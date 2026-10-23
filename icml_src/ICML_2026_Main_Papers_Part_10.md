# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 10 of 197

## 目录 (Table of Contents)

1. [OC-space: a Unifying Perspective on Verification of Tree Ensembles](#paper-0298)
2. [Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks](#paper-0299)
3. [Walrus: A Cross-domain Foundation Model for Continuum Dynamics](#paper-0300)
4. [PhenoBrain: Phenotype-Conditioned Long-Range Communication for Multi-Modal Brain Network Analysis](#paper-0301)
5. [How Many Different Outputs Can a Transformer Generate?](#paper-0302)
6. [Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](#paper-0303)
7. [Understanding Self-Supervised Learning via Latent Distribution Matching](#paper-0304)
8. [Emergent Analogical Reasoning in Transformers](#paper-0305)
9. [Rational Transductors](#paper-0306)
10. [How much can language models memorize?](#paper-0307)
11. [Post-Training with Policy Gradients: Optimality and the Base Model Barrier](#paper-0308)
12. [On the Convergence Rate of LoRA Gradient Descent](#paper-0309)
13. [Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs](#paper-0310)
14. [Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity](#paper-0311)
15. [Flowers: A Warp Drive for Neural PDE Solvers](#paper-0312)
16. [Accelerating Q-learning through Efficient Value-Sharing across Actions](#paper-0313)
17. [HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks](#paper-0314)
18. [MuonSSM: Orthogonalizing State Space Models for Sequence Modeling](#paper-0315)
19. [Exact Unlearning in Reinforcement Learning](#paper-0316)
20. [The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models](#paper-0317)
21. [Loss-Aware Distributionally Robust Optimization via Trainable Optimal Transport Ambiguity Sets](#paper-0318)
22. [Linguistic Properties and Model Scale in Brain Encoding: From Small to Compressed Language Models](#paper-0319)
23. [Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities](#paper-0320)
24. [Measuring Agents in Production](#paper-0321)
25. [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](#paper-0322)
26. [Adaptive Policy Backbone via Shared Network](#paper-0323)
27. [On Minimum Depth and Width of Floating-Point Neural Networks for Representing Floating-Point Functions](#paper-0324)
28. [Stable Deep Reinforcement Learning via Isotropic Gaussian Representations](#paper-0325)
29. [SoftJAX & SoftTorch: Empowering Automatic Differentiation Libraries with Informative Gradients](#paper-0326)
30. [Safety Alignment of LMs via Non-cooperative Games](#paper-0327)
31. [Recurrent Equivariant Constraint Modulation: Learning Per-Layer Symmetry Relaxation from Data](#paper-0328)
32. [Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment](#paper-0329)
33. [BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs](#paper-0330)

---


<a id="paper-0298"></a>

## OC-space: a Unifying Perspective on Verification of Tree Ensembles

### Images

![004ec7e7a26a1f3b1b6b2d1af87629ba1a7ec7742a9908df29662674c4a5ee05.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/004ec7e7a26a1f3b1b6b2d1af87629ba1a7ec7742a9908df29662674c4a5ee05.jpg)

![18746a0b030f0ad2013b62680cfda987a301fd850b7c05abab239e25785f45b9.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/18746a0b030f0ad2013b62680cfda987a301fd850b7c05abab239e25785f45b9.jpg)

![1951bdf4c48433503c3e5ea5ae9dd4c3dc95eaf6706aa2fc82e729196f06da77.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/1951bdf4c48433503c3e5ea5ae9dd4c3dc95eaf6706aa2fc82e729196f06da77.jpg)

![375acb25c6d03a5bde8b96c4be83a3e18a5eb046a009916ee2f912db72effca2.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/375acb25c6d03a5bde8b96c4be83a3e18a5eb046a009916ee2f912db72effca2.jpg)

![8c7d475cca4aec9b1b8b050a06492a3474fde2f92333d3036adaab016ec231a5.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/8c7d475cca4aec9b1b8b050a06492a3474fde2f92333d3036adaab016ec231a5.jpg)

![b1df34d397f53d13fbc78689743423ed44b890bad9038e552fe9ec4c0e2f27b9.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/b1df34d397f53d13fbc78689743423ed44b890bad9038e552fe9ec4c0e2f27b9.jpg)

![ba61024b33f2bcf0b865c2523b451425d9ef562ee7838f6b3d62826e3c68c05a.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/ba61024b33f2bcf0b865c2523b451425d9ef562ee7838f6b3d62826e3c68c05a.jpg)

![ceb8670055ef281b0864d2e9f0adf97ff44c1a95815d74f7441eccb007d8291b.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/ceb8670055ef281b0864d2e9f0adf97ff44c1a95815d74f7441eccb007d8291b.jpg)

![d0960d60c67f8b0cab0c9365fe12ab26b3e47b465823de45380eddc601c22ceb.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/images/d0960d60c67f8b0cab0c9365fe12ab26b3e47b465823de45380eddc601c22ceb.jpg)

### Tables

![022131925469533eccf8c461cb9e6630612fa44e626d4974b0ffc40bed47096b.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/022131925469533eccf8c461cb9e6630612fa44e626d4974b0ffc40bed47096b.jpg)

![13594ec21552532815121155bb6e987d2f7f963195f0ce70162eb6d40e317afb.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/13594ec21552532815121155bb6e987d2f7f963195f0ce70162eb6d40e317afb.jpg)

![3f49be4f3710085fbd5c4a3e3eae0cc6b730458673885eea5e14899f6b9d4c6e.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/3f49be4f3710085fbd5c4a3e3eae0cc6b730458673885eea5e14899f6b9d4c6e.jpg)

![4502ec05ce0d7d50a5f70fd89a2f60e7784557c8756ecf4f62774bff57f030cb.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/4502ec05ce0d7d50a5f70fd89a2f60e7784557c8756ecf4f62774bff57f030cb.jpg)

![6394c348edcfbe010707d4ae08f10880d294b49bcacc85bcadd5f8b434ebf540.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/6394c348edcfbe010707d4ae08f10880d294b49bcacc85bcadd5f8b434ebf540.jpg)

![73ab345184383b3f4bfd8150cb57615e81a345f16686bd069d169d8b3d68757d.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/73ab345184383b3f4bfd8150cb57615e81a345f16686bd069d169d8b3d68757d.jpg)

![7520907135681e56e45cedc97ee6b352240acd9288b0cf190e370cfe60e0a6aa.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/7520907135681e56e45cedc97ee6b352240acd9288b0cf190e370cfe60e0a6aa.jpg)

![7a8e224dbfb3b1e37f6f7ca29f17946d4ff3c6841582ce01874a75eafee10bb5.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/7a8e224dbfb3b1e37f6f7ca29f17946d4ff3c6841582ce01874a75eafee10bb5.jpg)

![a274d990744902aa502b278d0274f2fa5b7f77152fff282fb6537928029dd363.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/a274d990744902aa502b278d0274f2fa5b7f77152fff282fb6537928029dd363.jpg)

![bb0245047aedac5a3950ecc5f7cdfc5c6f0888467cd66c389ba7483f6f7dc7c6.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/bb0245047aedac5a3950ecc5f7cdfc5c6f0888467cd66c389ba7483f6f7dc7c6.jpg)

![d05d3e9e88ba687ca6f7997fb1eb97dc48394899c642488a4769368a691bc962.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/d05d3e9e88ba687ca6f7997fb1eb97dc48394899c642488a4769368a691bc962.jpg)

![e1c716c9cf79f4ca9bf07c9e21377557a36d0b9edd51df453154c7ed56a03d1a.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/e1c716c9cf79f4ca9bf07c9e21377557a36d0b9edd51df453154c7ed56a03d1a.jpg)

![f7136db808069c49ff7c2135633ea3dda4ede41b1969a61c295d063329c14da3.jpg](../icml_results/302_Benchmarking%20at%20the%20Edge%20of%20Comprehension/tables/f7136db808069c49ff7c2135633ea3dda4ede41b1969a61c295d063329c14da3.jpg)

## OC-space: a Unifying Perspective on Verification of Tree Ensembles


### Images

![1ebf92d24c0bcf7d86444eee78238e70f5a1f9d2fd223ddda271b4cef96862d5.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/1ebf92d24c0bcf7d86444eee78238e70f5a1f9d2fd223ddda271b4cef96862d5.jpg)

![24a34e2353915267674ef86c2e4fda01e6b34001ea6340a93c29ded69dfe9d7b.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/24a34e2353915267674ef86c2e4fda01e6b34001ea6340a93c29ded69dfe9d7b.jpg)

![5cd39dae0685e7cd509df647a1d8841bba99bdf86eaa3c745d46b3f1a33184ee.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/5cd39dae0685e7cd509df647a1d8841bba99bdf86eaa3c745d46b3f1a33184ee.jpg)

![956f98cce0a5ed58412b7287d783b6cb9400037603ef562a14f39beca907e3c5.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/956f98cce0a5ed58412b7287d783b6cb9400037603ef562a14f39beca907e3c5.jpg)

![9a5b68612ea9b4abd131270247c9fff129a26b3e399ffa5b23e65959a5d20ce6.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/9a5b68612ea9b4abd131270247c9fff129a26b3e399ffa5b23e65959a5d20ce6.jpg)

![b8c87e976baea573ce7b379f6ecf3160bac761a22f64e48dfa4e87a6bf17cab2.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/b8c87e976baea573ce7b379f6ecf3160bac761a22f64e48dfa4e87a6bf17cab2.jpg)

![bc829626e849cd8ae82a542df4e71157527a880104f29a4a59d79f816745aeb3.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/bc829626e849cd8ae82a542df4e71157527a880104f29a4a59d79f816745aeb3.jpg)

![e25813c5856dca176d0cefd36af5dd65e22d0dea8fa483f0aa544da46aa48ce7.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/images/e25813c5856dca176d0cefd36af5dd65e22d0dea8fa483f0aa544da46aa48ce7.jpg)

### Tables

![2f5a578ab87657e36b0e4d47bb01256e75a942e7f4b4cf1c9e5d6bc03843f5ab.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/2f5a578ab87657e36b0e4d47bb01256e75a942e7f4b4cf1c9e5d6bc03843f5ab.jpg)

![3a91a23d087fc674d144b6f76e86cf128f10278d46c8ed4b19ef43e277953eff.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/3a91a23d087fc674d144b6f76e86cf128f10278d46c8ed4b19ef43e277953eff.jpg)

![52e1d6ba2714920dd6b76f4688c11dac78fdeb560c9e5b94778838c2bb71a61d.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/52e1d6ba2714920dd6b76f4688c11dac78fdeb560c9e5b94778838c2bb71a61d.jpg)

![689f7cf43cd33b9957af9deac5754724f584c52c60be4dd8b37885a11198935e.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/689f7cf43cd33b9957af9deac5754724f584c52c60be4dd8b37885a11198935e.jpg)

![7338255e9fd1fd0e820ec51f927649c2d28fc9275a75097aa1725df23a9529f7.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/7338255e9fd1fd0e820ec51f927649c2d28fc9275a75097aa1725df23a9529f7.jpg)

![81b1aa8cba886085baf5f609021e0e3aeb4544fe8f1da71b54f00425e43d0851.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/81b1aa8cba886085baf5f609021e0e3aeb4544fe8f1da71b54f00425e43d0851.jpg)

![9f2704f600fd582d6a94106f987b76c0ab1eb1e8edcbbe6612f61d7075eef089.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/9f2704f600fd582d6a94106f987b76c0ab1eb1e8edcbbe6612f61d7075eef089.jpg)

![d3e71fb0b5472ed2400875282e9a8a81cef32a0831d5680f09e00411a520623f.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/d3e71fb0b5472ed2400875282e9a8a81cef32a0831d5680f09e00411a520623f.jpg)

![d64e79726fa08592e0075a2efd524aa3b7aba288e4e473bc59cdf3353049e345.jpg](../icml_results/303_OC-space_%20a%20Unifying%20Perspective%20on%20Verification%20of%20Tree%20Ensembles/tables/d64e79726fa08592e0075a2efd524aa3b7aba288e4e473bc59cdf3353049e345.jpg)

<a id="paper-0299"></a>

## Foundations of Equivariant Deep Learning: Unifying Graph and Sheaf Neural Networks


### Tables

![4709d741ea4cc9675c64a4d895e84cd9536b402e9d3067275e4e06ac170a513a.jpg](../icml_results/304_Foundations%20of%20Equivariant%20Deep%20Learning_%20Unifying%20Graph%20and%20Sheaf%20Neural%20Networks/tables/4709d741ea4cc9675c64a4d895e84cd9536b402e9d3067275e4e06ac170a513a.jpg)

![b6cddc8ab47f655ca7cf5afda4ed5eff75411aa5647195698a1f080f8ca6a747.jpg](../icml_results/304_Foundations%20of%20Equivariant%20Deep%20Learning_%20Unifying%20Graph%20and%20Sheaf%20Neural%20Networks/tables/b6cddc8ab47f655ca7cf5afda4ed5eff75411aa5647195698a1f080f8ca6a747.jpg)

<a id="paper-0300"></a>

## Walrus: A Cross-domain Foundation Model for Continuum Dynamics


### Images

![0642f304493340ae18d656af66a1fca1f242fb414bc360c7b32721fcf96b9e4b.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/0642f304493340ae18d656af66a1fca1f242fb414bc360c7b32721fcf96b9e4b.jpg)

![0fb09ff574b7094bee228f1cfc236ed73dc52c2243375f9799603277fe858d26.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/0fb09ff574b7094bee228f1cfc236ed73dc52c2243375f9799603277fe858d26.jpg)

![130739a9f1197e82f12cd377927f735ad4409391399de6ee56987dbe0da9f5c0.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/130739a9f1197e82f12cd377927f735ad4409391399de6ee56987dbe0da9f5c0.jpg)

![1a0f3c4b39d4f1ecfc92e81ec4e1325e505ab445c4b7297b9a6fbd6faef96f0f.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/1a0f3c4b39d4f1ecfc92e81ec4e1325e505ab445c4b7297b9a6fbd6faef96f0f.jpg)

![2c709818bd2f4dfc47f6d14e2701104c8db64a2917a68d957020fce04f8e4010.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/2c709818bd2f4dfc47f6d14e2701104c8db64a2917a68d957020fce04f8e4010.jpg)

![2dba29820b1e7fb172962791fa6fadf5fef0f0f3631f9c773c5bf639fe63d2b4.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/2dba29820b1e7fb172962791fa6fadf5fef0f0f3631f9c773c5bf639fe63d2b4.jpg)

![4b24b21dc509651ce89afb8f651511270ac7b6c3784f7febaec26e6ef6d194a9.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/4b24b21dc509651ce89afb8f651511270ac7b6c3784f7febaec26e6ef6d194a9.jpg)

![4d0417d2c9fa848649b9334da04cfe8b1006f26952db56deef04db4d14f57e85.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/4d0417d2c9fa848649b9334da04cfe8b1006f26952db56deef04db4d14f57e85.jpg)

![536387f1a7c72cde69fac155204c8c7771e47e215bc9fd0d58d16e12e8d68c98.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/536387f1a7c72cde69fac155204c8c7771e47e215bc9fd0d58d16e12e8d68c98.jpg)

![53d75fc455dce579021c7f0e936d065bef896edb0b2ad97e8938882dc5c9dd5a.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/53d75fc455dce579021c7f0e936d065bef896edb0b2ad97e8938882dc5c9dd5a.jpg)

![5a356574b393d567cd11801dc3637e8806df4552cae4802a244afa2d9a688953.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/5a356574b393d567cd11801dc3637e8806df4552cae4802a244afa2d9a688953.jpg)

![5a7dfa1c724775bb6559fd9a8c93666edb4b26cab4135a75a8b04c92cf3a2a43.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/5a7dfa1c724775bb6559fd9a8c93666edb4b26cab4135a75a8b04c92cf3a2a43.jpg)

![5ef2bdc64429ff535bcda4786a034920ef0f0a91768ef83dce5af8c4dadd83de.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/5ef2bdc64429ff535bcda4786a034920ef0f0a91768ef83dce5af8c4dadd83de.jpg)

![69607b22ed2f003bf37b1c457bfa9379f50d00fe36c4960dd42c46127eb93c71.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/69607b22ed2f003bf37b1c457bfa9379f50d00fe36c4960dd42c46127eb93c71.jpg)

![6a4a1a90b136ee62c7b34571f538e5d29829dca7dd92b9db0d83e42ba72d7d2c.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/6a4a1a90b136ee62c7b34571f538e5d29829dca7dd92b9db0d83e42ba72d7d2c.jpg)

![6cf580c0be409d4140741eee7a53f78a24bdb7d5b49e46662ab7ccb193ca0670.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/6cf580c0be409d4140741eee7a53f78a24bdb7d5b49e46662ab7ccb193ca0670.jpg)

![762b95a2abec62f5e547cf03c00c763626130d76680f1275dd0b254accfeec51.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/762b95a2abec62f5e547cf03c00c763626130d76680f1275dd0b254accfeec51.jpg)

![78bfa586a3da2c7efcdea6c86f6d040291081bec949dedf13f8fdc5b30ae1439.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/78bfa586a3da2c7efcdea6c86f6d040291081bec949dedf13f8fdc5b30ae1439.jpg)

![89aec8a6162efa9af8b44979d388108fc74ce4cff1ee3c2fd28c4fdb977baaaf.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/89aec8a6162efa9af8b44979d388108fc74ce4cff1ee3c2fd28c4fdb977baaaf.jpg)

![8c4e98ce92515dacfb5110c203749138981eba942a5c45b63dd085bebf84e119.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/8c4e98ce92515dacfb5110c203749138981eba942a5c45b63dd085bebf84e119.jpg)

![8c95aa51aa39961889fe43a26479fe9214a77ba496691cc9b90c186f75794427.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/8c95aa51aa39961889fe43a26479fe9214a77ba496691cc9b90c186f75794427.jpg)

![8ee77d9768dfb6ff0f1715d085c69d18aac293051db56de82b2924b8b337b23c.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/8ee77d9768dfb6ff0f1715d085c69d18aac293051db56de82b2924b8b337b23c.jpg)

![927ba95a312eacc52a1848e1e014ebe5d7299239790ad3a7653b8017f0bda829.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/927ba95a312eacc52a1848e1e014ebe5d7299239790ad3a7653b8017f0bda829.jpg)

![a7074c0a13b20a1dd532ebc4c7f378ea421b93c079f7366be6397521360bed74.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/a7074c0a13b20a1dd532ebc4c7f378ea421b93c079f7366be6397521360bed74.jpg)

![a915db95effb1edf4596be8a086400312c29f0c33a3c8b3331888287be3b3c21.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/a915db95effb1edf4596be8a086400312c29f0c33a3c8b3331888287be3b3c21.jpg)

![b09a2145ffce56fd4fa83031a135f8dc1bb099291aa1385653d1508726dd84f1.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/b09a2145ffce56fd4fa83031a135f8dc1bb099291aa1385653d1508726dd84f1.jpg)

![b489d1590e257503ce510b6857863dcef70737a2d160a1a544f44812dc5a8773.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/b489d1590e257503ce510b6857863dcef70737a2d160a1a544f44812dc5a8773.jpg)

![bdffe55794c2e26470d7bc61e4ea1a5b22cdefe2a1f475aea75209e5001fa6cd.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/bdffe55794c2e26470d7bc61e4ea1a5b22cdefe2a1f475aea75209e5001fa6cd.jpg)

![be5af1a944c3dad8a6c83d19a40acabc6f4b90d8401e1c75315ed8bfa2fb0128.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/be5af1a944c3dad8a6c83d19a40acabc6f4b90d8401e1c75315ed8bfa2fb0128.jpg)

![beab76660c580314b3b9f8134126f28ea95fd65c5efed9eb4c5cc08292538205.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/beab76660c580314b3b9f8134126f28ea95fd65c5efed9eb4c5cc08292538205.jpg)

![c3ec6dd2298e901c410d23d7f636d9d84ee179808dc24b2ac9071b3d4e976c0d.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/c3ec6dd2298e901c410d23d7f636d9d84ee179808dc24b2ac9071b3d4e976c0d.jpg)

![c6f7b41f4c29f580a5f06cbcc7f058521885d9969130bc693d55e29a22cf36b0.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/c6f7b41f4c29f580a5f06cbcc7f058521885d9969130bc693d55e29a22cf36b0.jpg)

![cfef516b8aec53921891475c445b69946cfd268a61956a50f5d8d53e2920016e.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/cfef516b8aec53921891475c445b69946cfd268a61956a50f5d8d53e2920016e.jpg)

![d1c436cb796ebadad4d73e99ea23704bd03108c026f4086e36322799681d7e62.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/d1c436cb796ebadad4d73e99ea23704bd03108c026f4086e36322799681d7e62.jpg)

![d29dec8f4311424a60e71190bc8857ebe34409f853d32697c94b78ce27629f02.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/d29dec8f4311424a60e71190bc8857ebe34409f853d32697c94b78ce27629f02.jpg)

![d9a2ba3760ce9232f7d1529cebcdf87c6abc02638f33e414021a985e6067d016.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/d9a2ba3760ce9232f7d1529cebcdf87c6abc02638f33e414021a985e6067d016.jpg)

![d9fd0a069e56ecd1f86b3d24e7e1658ec330324995ed4dbbcb994cea9d7002e8.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/d9fd0a069e56ecd1f86b3d24e7e1658ec330324995ed4dbbcb994cea9d7002e8.jpg)

![e39834d0c3cf8f4c451db62e506058eec1eea15d25dca7704a9cae04700b363a.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/e39834d0c3cf8f4c451db62e506058eec1eea15d25dca7704a9cae04700b363a.jpg)

![e78b710fcfac68a8eadc542d332da5815c5c82386eb9b1fefe5004bb7e3a356f.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/e78b710fcfac68a8eadc542d332da5815c5c82386eb9b1fefe5004bb7e3a356f.jpg)

![ecaf4c4c8b54d43577c37d36183de84db9311ef87c6f0eee9586c2b50089058c.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/ecaf4c4c8b54d43577c37d36183de84db9311ef87c6f0eee9586c2b50089058c.jpg)

![f03e1bbdad460695c1e731f23672ff4480e3571bf636028bea4eefe8c3bc781d.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/f03e1bbdad460695c1e731f23672ff4480e3571bf636028bea4eefe8c3bc781d.jpg)

![fd8e7a7a826a92d0767d5db0d167166b177c936ffc425c7fe23de7ca41128fb6.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/images/fd8e7a7a826a92d0767d5db0d167166b177c936ffc425c7fe23de7ca41128fb6.jpg)

### Tables

![045f990d3b16917582c5135ba1ee31f0da1a2606c71d84cea42790f1f4f2fe07.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/045f990d3b16917582c5135ba1ee31f0da1a2606c71d84cea42790f1f4f2fe07.jpg)

![2281f3d66a9e45acc4ed24fec546929e6b43175c29b204f73759285b8026b8ce.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/2281f3d66a9e45acc4ed24fec546929e6b43175c29b204f73759285b8026b8ce.jpg)

![25ca768ebfdaed3f885045acb9272010c6a18105eb2681001beb6040e3fbcf98.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/25ca768ebfdaed3f885045acb9272010c6a18105eb2681001beb6040e3fbcf98.jpg)

![25e3a7c77e7e91ad8b26a1d0492a6ab5e3341010f1135fd166a58f0182d2b193.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/25e3a7c77e7e91ad8b26a1d0492a6ab5e3341010f1135fd166a58f0182d2b193.jpg)

![2df0b4645223b2324387aa688806a565f7a02711992b702e8c3416a640bc849b.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/2df0b4645223b2324387aa688806a565f7a02711992b702e8c3416a640bc849b.jpg)

![3a86f14dee35b9642259cbe75f27ab4e294dee2e12cfeea306d620b3f4a76fe3.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/3a86f14dee35b9642259cbe75f27ab4e294dee2e12cfeea306d620b3f4a76fe3.jpg)

![3eeff794b1adbfd60ee143afd320f060062c9d2e14b3f4384b1933dbcbecceec.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/3eeff794b1adbfd60ee143afd320f060062c9d2e14b3f4384b1933dbcbecceec.jpg)

![48d2ab504de3da2bd30dccdc7ca7cae2738cf0bb28c306ed0e19bbfc5d3740e0.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/48d2ab504de3da2bd30dccdc7ca7cae2738cf0bb28c306ed0e19bbfc5d3740e0.jpg)

![4e10addfee068752b5f9063661a352da004db428279c5651055dbd4478311280.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/4e10addfee068752b5f9063661a352da004db428279c5651055dbd4478311280.jpg)

![6695d41f14f2ac3d350a89888b5e8fde6055a77b6444c4274c1ec246203fc78d.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/6695d41f14f2ac3d350a89888b5e8fde6055a77b6444c4274c1ec246203fc78d.jpg)

![69138a3bf4951806103bdbf30a24d3495a4dc97e3c2892914267ed99bdf5ad29.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/69138a3bf4951806103bdbf30a24d3495a4dc97e3c2892914267ed99bdf5ad29.jpg)

![7607d38396200f3a729190a758151bb620c984cc4fbb32807e3b347f7ffa6fbf.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/7607d38396200f3a729190a758151bb620c984cc4fbb32807e3b347f7ffa6fbf.jpg)

![8f0f08b207bd6eb4a071bf02820fa0b0a69f1e8bc92ae60a5058e2adc492859b.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/8f0f08b207bd6eb4a071bf02820fa0b0a69f1e8bc92ae60a5058e2adc492859b.jpg)

![940314ac9c332ba486d209530af7451412210c40ff53d6095721bb19ec7232ce.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/940314ac9c332ba486d209530af7451412210c40ff53d6095721bb19ec7232ce.jpg)

![9919a808a52ac83f7122df7980bc774e1f17ad1000844ad64886c26ba0f018b7.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/9919a808a52ac83f7122df7980bc774e1f17ad1000844ad64886c26ba0f018b7.jpg)

![a9b8cfa62346860b2a84ddd39777223ec7a6622c17e1005865d2e110384ec8c1.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/a9b8cfa62346860b2a84ddd39777223ec7a6622c17e1005865d2e110384ec8c1.jpg)

![e4f8b8f1f68a04d5d86bca8650597335137412d954ece6b2f31dab8680af5cf3.jpg](../icml_results/305_Walrus_%20A%20Cross-domain%20Foundation%20Model%20for%20Continuum%20Dynamics/tables/e4f8b8f1f68a04d5d86bca8650597335137412d954ece6b2f31dab8680af5cf3.jpg)

<a id="paper-0301"></a>

## PhenoBrain: Phenotype-Conditioned Long-Range Communication for Multi-Modal Brain Network Analysis


### Images

![08b7e6ca1b39ae2e6fd5958ec2bd7304080429fd46a15c31fb341e0dc53702f7.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/08b7e6ca1b39ae2e6fd5958ec2bd7304080429fd46a15c31fb341e0dc53702f7.jpg)

![2039634f3f914692bfa6121871e44b24f24f000e8c0923a0019983273559b678.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/2039634f3f914692bfa6121871e44b24f24f000e8c0923a0019983273559b678.jpg)

![24c131f0a2f119c4ea9dcfd9e5066f2da30fca2de666ed44bd0fb4f976447d8d.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/24c131f0a2f119c4ea9dcfd9e5066f2da30fca2de666ed44bd0fb4f976447d8d.jpg)

![305031d77844cd3ee7ef7029c144cfaf02b6ebde1a40ed44a1081c8e73cd90d1.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/305031d77844cd3ee7ef7029c144cfaf02b6ebde1a40ed44a1081c8e73cd90d1.jpg)

![a0a51fe3469c8f1d21b9680b1ae2f1be094fd9498388d675a38551978a6d0271.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/a0a51fe3469c8f1d21b9680b1ae2f1be094fd9498388d675a38551978a6d0271.jpg)

![ac633c6b7eba83226d68a682eba75b2fbe156e90fa47ecd500c5e99290726820.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/ac633c6b7eba83226d68a682eba75b2fbe156e90fa47ecd500c5e99290726820.jpg)

![b2dbc74fc2b3f1c17bbac322c6e09a71805c096ebf324caa6ca1b00424697f87.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/b2dbc74fc2b3f1c17bbac322c6e09a71805c096ebf324caa6ca1b00424697f87.jpg)

![c29c09cdbb99a3198b9c192085e8091bd9d16877335248c1fbfe5eefc9036318.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/c29c09cdbb99a3198b9c192085e8091bd9d16877335248c1fbfe5eefc9036318.jpg)

![f62988f09afb0e495d50b06280e1379749c3c19e73a0bf852f5e55bc5d232a4e.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/images/f62988f09afb0e495d50b06280e1379749c3c19e73a0bf852f5e55bc5d232a4e.jpg)

### Tables

![5bd6e207c27a75db2e23bf52cf9175a36ecc708a46b8c87380fd116a0824a83e.jpg](../icml_results/306_PhenoBrain_%20Phenotype-Conditioned%20Long-Range%20Communication%20for%20Multi-Modal%20Brain%20Network%20Analysis/tables/5bd6e207c27a75db2e23bf52cf9175a36ecc708a46b8c87380fd116a0824a83e.jpg)

<a id="paper-0302"></a>

## How Many Different Outputs Can a Transformer Generate?


### Images

![0082711946ff6ab914512ecbe2425ae68abd65c796c4c228ea1fefa951b56fb6.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/0082711946ff6ab914512ecbe2425ae68abd65c796c4c228ea1fefa951b56fb6.jpg)

![0ac1032e4ab90b4327f43d31d0e4041d0c3eefe23b9333830d5274947c8d21cf.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/0ac1032e4ab90b4327f43d31d0e4041d0c3eefe23b9333830d5274947c8d21cf.jpg)

![61d08853386b42a3c3a28552f96d47de9b03cccff3607a827b0244d6f77e0f2f.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/61d08853386b42a3c3a28552f96d47de9b03cccff3607a827b0244d6f77e0f2f.jpg)

![61dd8f11b23aede7155fbd94499a497e4e80d7de1ba204ffb1e8a13e6d5700c0.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/61dd8f11b23aede7155fbd94499a497e4e80d7de1ba204ffb1e8a13e6d5700c0.jpg)

![70b790cff3d92ff6d5fbe0da115897fe070272545144a3db0633ed8f93bfdaeb.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/70b790cff3d92ff6d5fbe0da115897fe070272545144a3db0633ed8f93bfdaeb.jpg)

![8231623ec05e449d919137f76780c3efe0a652c25386b9bb18717abbdd692671.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/8231623ec05e449d919137f76780c3efe0a652c25386b9bb18717abbdd692671.jpg)

![85e6860d09d46b8e57eda9687e0a3357d700ae9567dacfdcd91ca4d19375b001.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/85e6860d09d46b8e57eda9687e0a3357d700ae9567dacfdcd91ca4d19375b001.jpg)

![a6133cb69f33b114e10df4c4187a9480d932e4828c1f5740cd94bbe79ae08adf.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/a6133cb69f33b114e10df4c4187a9480d932e4828c1f5740cd94bbe79ae08adf.jpg)

![d6482f7ea46874d87cf20bef358948c549c0eeace2fa9bed3c8f5b5e169e9282.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/d6482f7ea46874d87cf20bef358948c549c0eeace2fa9bed3c8f5b5e169e9282.jpg)

![ef961eb2923790333eccafc58554d5ff206e77df9e294d855f24b1226e902f2a.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/ef961eb2923790333eccafc58554d5ff206e77df9e294d855f24b1226e902f2a.jpg)

![fde195a1795f69a818912a175adaeaf3e01ac61878f1f84e84140df944d6aabb.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/images/fde195a1795f69a818912a175adaeaf3e01ac61878f1f84e84140df944d6aabb.jpg)

### Tables

![1c56b7a533cb539dae69eb79af3ecc45d91eb114e6b77116fd6ebe79f5c8eed4.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/tables/1c56b7a533cb539dae69eb79af3ecc45d91eb114e6b77116fd6ebe79f5c8eed4.jpg)

![247ae71d2d3ffb3b4a9a30496cd215bb5765af6b4c42ea6e2f2f3f7cefc52751.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/tables/247ae71d2d3ffb3b4a9a30496cd215bb5765af6b4c42ea6e2f2f3f7cefc52751.jpg)

![7d9efe60d7c193012e8474466fb0aec37f8635ee4b8556cb0cfa39395149d774.jpg](../icml_results/307_How%20Many%20Different%20Outputs%20Can%20a%20Transformer%20Generate_/tables/7d9efe60d7c193012e8474466fb0aec37f8635ee4b8556cb0cfa39395149d774.jpg)

<a id="paper-0303"></a>

## Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents


### Images

![114640034068ec1112fe13b10bf7b3cc669c9d4886a792348382b6ee57a41e2f.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/images/114640034068ec1112fe13b10bf7b3cc669c9d4886a792348382b6ee57a41e2f.jpg)

![40592621def6f5ba835b3aadbb620b422699efb003c4e1c04287a834c966ac71.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/images/40592621def6f5ba835b3aadbb620b422699efb003c4e1c04287a834c966ac71.jpg)

![53633864371618e0fc2d03aa1f63e8d6a4e6ab92a953748f07be50b2525d5508.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/images/53633864371618e0fc2d03aa1f63e8d6a4e6ab92a953748f07be50b2525d5508.jpg)

![5cf85603e808569a858771b345caa8de547e623f59c67298a9b5e147fa5577a7.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/images/5cf85603e808569a858771b345caa8de547e623f59c67298a9b5e147fa5577a7.jpg)

![9f6f987eb14de64ed1a95287519ee59889d4a0ee80756857b9ee2e6c70cbffce.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/images/9f6f987eb14de64ed1a95287519ee59889d4a0ee80756857b9ee2e6c70cbffce.jpg)

![ae79a9eed7fd2974fe1dbd0b81c84d815c62b6c972150358f359754e3620433d.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/images/ae79a9eed7fd2974fe1dbd0b81c84d815c62b6c972150358f359754e3620433d.jpg)

### Tables

![1f43e66fa49929ccb3a940123e6b197c1980cae7d15c4f5f1d5d5026e464b413.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/tables/1f43e66fa49929ccb3a940123e6b197c1980cae7d15c4f5f1d5d5026e464b413.jpg)

![55cf37870ee3c66e14e6caa498cc7ec1b53251e295523dca07d42624e64a4ce8.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/tables/55cf37870ee3c66e14e6caa498cc7ec1b53251e295523dca07d42624e64a4ce8.jpg)

![5c8c5ec4b0fcd161fe454c55645c778789f24866362806c9bda2fad4965f9779.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/tables/5c8c5ec4b0fcd161fe454c55645c778789f24866362806c9bda2fad4965f9779.jpg)

![6a7a83eb81e8a6a8b988d71fffd560f1042b0a6ec74b87d8257c7e5258fd4d34.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/tables/6a7a83eb81e8a6a8b988d71fffd560f1042b0a6ec74b87d8257c7e5258fd4d34.jpg)

![941a15cc2fedc99b6cc20f324bc766d24c1d2718d76739aaeb690a7cb30eafdf.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/tables/941a15cc2fedc99b6cc20f324bc766d24c1d2718d76739aaeb690a7cb30eafdf.jpg)

![e3300d03c494c2004cbcc40b4129b7c5e3193d4e62bcaad4df405d6cd6a6a81c.jpg](../icml_results/308_Skill-Pro_%20Learning%20Reusable%20Skills%20from%20Experience%20via%20Non-Parametric%20PPO%20for%20LLM%20Agents/tables/e3300d03c494c2004cbcc40b4129b7c5e3193d4e62bcaad4df405d6cd6a6a81c.jpg)

<a id="paper-0304"></a>

## Understanding Self-Supervised Learning via Latent Distribution Matching


### Images

![099701f71aabee2c6e54aee87e1af671a2b6dbecfe3daceb0c9762425e616d4c.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/099701f71aabee2c6e54aee87e1af671a2b6dbecfe3daceb0c9762425e616d4c.jpg)

![0c7c0df5686a594c720d99e82a0d99061dca6fbb711918861f8be7ed96276132.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/0c7c0df5686a594c720d99e82a0d99061dca6fbb711918861f8be7ed96276132.jpg)

![25d4b613a064b57273004f0910245f04316ef9d98e44c6d0998ab11d139f890b.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/25d4b613a064b57273004f0910245f04316ef9d98e44c6d0998ab11d139f890b.jpg)

![30fcdf8aa7f109c45c820d1ac5a760a0ee01f99f5f6d4d788090490a56adc939.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/30fcdf8aa7f109c45c820d1ac5a760a0ee01f99f5f6d4d788090490a56adc939.jpg)

![43e50304d3918cc13888b466d08b5a26c5cba0aff69a6c77f094664daf782114.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/43e50304d3918cc13888b466d08b5a26c5cba0aff69a6c77f094664daf782114.jpg)

![57e18fd7d4e5a67633ad9fb42d2283810f239c5932a62dbaa5057141126a6d7b.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/57e18fd7d4e5a67633ad9fb42d2283810f239c5932a62dbaa5057141126a6d7b.jpg)

![73b97787d54a128bdb76b153d4dd7a88c51c75119c18c5d1d9ebea80171daa5a.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/73b97787d54a128bdb76b153d4dd7a88c51c75119c18c5d1d9ebea80171daa5a.jpg)

![7be2efc45bdb505e32a06caa61380fca0a2af08727b1f641b7dc5e744b969c3e.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/7be2efc45bdb505e32a06caa61380fca0a2af08727b1f641b7dc5e744b969c3e.jpg)

![d750b11282e7386ed0fb791039233d0c5cdfb1f4d912adb07a70cbc38d8ec869.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/d750b11282e7386ed0fb791039233d0c5cdfb1f4d912adb07a70cbc38d8ec869.jpg)

![d7aa3b12a933775b4a2df99b17a84a996273cf7d583d2e2dcc03aa8984848bbd.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/d7aa3b12a933775b4a2df99b17a84a996273cf7d583d2e2dcc03aa8984848bbd.jpg)

![db04e32b5ad31f262dc27aa95f264c844823c67cb4282aeb425c888b89f5221d.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/db04e32b5ad31f262dc27aa95f264c844823c67cb4282aeb425c888b89f5221d.jpg)

![df6a57399b6b62a22036b4017698546cc63b3186a22ae9b524df3dd8b57a86d3.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/df6a57399b6b62a22036b4017698546cc63b3186a22ae9b524df3dd8b57a86d3.jpg)

![e96320106ae7c3772518db16065d96791ddd174165745d88a1762e1b5c3090c5.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/e96320106ae7c3772518db16065d96791ddd174165745d88a1762e1b5c3090c5.jpg)

![ff117e998dc393bb03d7b67b71f005f53197016a58b5f5ebeb5accea4b3eea9f.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/images/ff117e998dc393bb03d7b67b71f005f53197016a58b5f5ebeb5accea4b3eea9f.jpg)

### Tables

![44ede016a79b37db5079220baf4b1373a61ac1c0d1a2e430a2c1ae139d4ab66f.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/tables/44ede016a79b37db5079220baf4b1373a61ac1c0d1a2e430a2c1ae139d4ab66f.jpg)

![4f84b2c5eb18178eaeaa64fd8543684cde6bd955f9e71fb52cd8a531a9183417.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/tables/4f84b2c5eb18178eaeaa64fd8543684cde6bd955f9e71fb52cd8a531a9183417.jpg)

![aa42885d21dd20d91df8c26594785681f624af6a1933b70d5fed9ca5349ac2ab.jpg](../icml_results/309_Understanding%20Self-Supervised%20Learning%20via%20Latent%20Distribution%20Matching/tables/aa42885d21dd20d91df8c26594785681f624af6a1933b70d5fed9ca5349ac2ab.jpg)

<a id="paper-0305"></a>

## Emergent Analogical Reasoning in Transformers


### Images

![0afd9160df498182fd0b03761a476a0aa774a3442bc8275a1f484c6a3e176778.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/0afd9160df498182fd0b03761a476a0aa774a3442bc8275a1f484c6a3e176778.jpg)

![0f116d22d9c979ac34157dd950b943095ff3a2464ec41a60b8ffa22569c6f31e.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/0f116d22d9c979ac34157dd950b943095ff3a2464ec41a60b8ffa22569c6f31e.jpg)

![19a8a54939f4407ea486551a783db0fb5f6553373fa9c849e5f88fd686b5e719.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/19a8a54939f4407ea486551a783db0fb5f6553373fa9c849e5f88fd686b5e719.jpg)

![265babae3f0507e0561d6c5535611ebbc43fed42ae6e0e39348045d80bd9cced.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/265babae3f0507e0561d6c5535611ebbc43fed42ae6e0e39348045d80bd9cced.jpg)

![356ac8cff7fcbf8f3a486184477fb77aba1a001d77943fff872766079cd1c714.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/356ac8cff7fcbf8f3a486184477fb77aba1a001d77943fff872766079cd1c714.jpg)

![3f0c5fed81192ab9c90f40d88e231bf908671caf2f38a2762a75b57df6e5b708.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/3f0c5fed81192ab9c90f40d88e231bf908671caf2f38a2762a75b57df6e5b708.jpg)

![53e10888790d1a41e7c08b33ce6867f91c4a9d732898d0631c30e3e3acbb9f41.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/53e10888790d1a41e7c08b33ce6867f91c4a9d732898d0631c30e3e3acbb9f41.jpg)

![582e2017350c1c91f26fd7d451cf98afa879ad231d0277374c6e6a33f2e70602.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/582e2017350c1c91f26fd7d451cf98afa879ad231d0277374c6e6a33f2e70602.jpg)

![61842b87361d9c69132064d48e5281bb3724b2db995769694451c1389a8a78ac.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/61842b87361d9c69132064d48e5281bb3724b2db995769694451c1389a8a78ac.jpg)

![727bb3c47c8da231f6eca776e53a7fe7dbd001d5a7e1373084e5011d5d6a079d.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/727bb3c47c8da231f6eca776e53a7fe7dbd001d5a7e1373084e5011d5d6a079d.jpg)

![7bef1f2e7f28ae08f753bebe8a6c3e30d33543239d4e9e7884bb4fb7f3ebdb48.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/7bef1f2e7f28ae08f753bebe8a6c3e30d33543239d4e9e7884bb4fb7f3ebdb48.jpg)

![7d0b9f122f52fe0e4a869abe8919a036391bf6c8af05c1f884791c1d34d41f47.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/7d0b9f122f52fe0e4a869abe8919a036391bf6c8af05c1f884791c1d34d41f47.jpg)

![8104d6f5b0c6a877dce38ff64b5a7ffe1440965bdc3b6aac3be4ea5e9f118ebd.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/8104d6f5b0c6a877dce38ff64b5a7ffe1440965bdc3b6aac3be4ea5e9f118ebd.jpg)

![837bf80df016fe74a820991e21ae58a7630f85b508551f9adba89bcfe4610bc3.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/837bf80df016fe74a820991e21ae58a7630f85b508551f9adba89bcfe4610bc3.jpg)

![84fb3d0c2d0605cdbca16d1a28f9e9c14c96d4bc1e90c1645b5f906125b66e2d.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/84fb3d0c2d0605cdbca16d1a28f9e9c14c96d4bc1e90c1645b5f906125b66e2d.jpg)

![94aa4b8de64fc3e23f99ecbde52b9e42a843fa3e45ef3f79eacacf298e4cfeb9.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/94aa4b8de64fc3e23f99ecbde52b9e42a843fa3e45ef3f79eacacf298e4cfeb9.jpg)

![9b617ed90ad8d79ffdff5095c599cbd629e7af4d4517b1135a8304b9261f6ea8.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/9b617ed90ad8d79ffdff5095c599cbd629e7af4d4517b1135a8304b9261f6ea8.jpg)

![a2c1e88118b6953d10e27ac54982a6ea405be3566c5bff6b40fa9c89d8e388b8.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/a2c1e88118b6953d10e27ac54982a6ea405be3566c5bff6b40fa9c89d8e388b8.jpg)

![a579af2e119f21b0a43d67f1a73abac878dd3738033e8776b3141bf7868c0ef2.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/a579af2e119f21b0a43d67f1a73abac878dd3738033e8776b3141bf7868c0ef2.jpg)

![a823c6573fc6d337e459fe544101d646ae5a161299c3ae65863620e8aad0a5e0.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/a823c6573fc6d337e459fe544101d646ae5a161299c3ae65863620e8aad0a5e0.jpg)

![b4fe28796a367d2795bf7d6f709899de148919b59cebc01916a334125d8c3167.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/b4fe28796a367d2795bf7d6f709899de148919b59cebc01916a334125d8c3167.jpg)

![c3e5b77bbac9fb7606373c252b110a90fdb7f39af026796e128680427fa06438.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/c3e5b77bbac9fb7606373c252b110a90fdb7f39af026796e128680427fa06438.jpg)

![c8e05c1355a9dc990b6d9af8b891e495cfdcc68747093fc5b7244be32f766b64.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/c8e05c1355a9dc990b6d9af8b891e495cfdcc68747093fc5b7244be32f766b64.jpg)

![d8fd95bc112303661b1977552829efe7d2c43b0846b20955019b2ef1e59360c7.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/d8fd95bc112303661b1977552829efe7d2c43b0846b20955019b2ef1e59360c7.jpg)

![debc062c5d28b611c74451882049e1074bd653099b07c5df3f4f94d4b5b5a846.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/debc062c5d28b611c74451882049e1074bd653099b07c5df3f4f94d4b5b5a846.jpg)

![ec13d70a4d34b78dffdad6fb169e5be987fe0cd9e4c60a50e9eecb029678a9f1.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/ec13d70a4d34b78dffdad6fb169e5be987fe0cd9e4c60a50e9eecb029678a9f1.jpg)

![f9de16dc2251e39112ff04b3a411efc8390621cbd2b94b390fca4c04d610589d.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/images/f9de16dc2251e39112ff04b3a411efc8390621cbd2b94b390fca4c04d610589d.jpg)

### Tables

![0424651d5248f92fd54cb6aefd7d1923b556ff2686d21d0025b3ac2f3d848874.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/0424651d5248f92fd54cb6aefd7d1923b556ff2686d21d0025b3ac2f3d848874.jpg)

![07864930f3ba56fe58de2a7916cd9a80273853ac00f9768442ad17c5fb3ba087.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/07864930f3ba56fe58de2a7916cd9a80273853ac00f9768442ad17c5fb3ba087.jpg)

![127ead24ad0ee4958b0d49d1b7969d022e583b9ba991f628e0343a4cc69dfafe.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/127ead24ad0ee4958b0d49d1b7969d022e583b9ba991f628e0343a4cc69dfafe.jpg)

![7313f16ed3ea0598b7d127558ba488276e2e79f10e28349b9f8dbea40b4d1033.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/7313f16ed3ea0598b7d127558ba488276e2e79f10e28349b9f8dbea40b4d1033.jpg)

![9a9956c0d960b21c3365b23ae35f00f174e09d1845963e79577d51ac8a1405fc.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/9a9956c0d960b21c3365b23ae35f00f174e09d1845963e79577d51ac8a1405fc.jpg)

![9e127d213a59c8628dcac78589366838c6e9f68c323ba9eac7c5b91163564e7f.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/9e127d213a59c8628dcac78589366838c6e9f68c323ba9eac7c5b91163564e7f.jpg)

![dd689cef5323917d9d2bab107d725f58d89b20e12899b2b5cdc7c981ba4f37c4.jpg](../icml_results/310_Emergent%20Analogical%20Reasoning%20in%20Transformers/tables/dd689cef5323917d9d2bab107d725f58d89b20e12899b2b5cdc7c981ba4f37c4.jpg)

<a id="paper-0306"></a>

## Rational Transductors


### Images

![151d37b4dd6ef4567a6ec8ff50091d48b5afdca458a44843b309a01252cc1710.jpg](../icml_results/311_Rational%20Transductors/images/151d37b4dd6ef4567a6ec8ff50091d48b5afdca458a44843b309a01252cc1710.jpg)

![160e55651ac5fa0b468081cd378b7503ec0af26c15fb4936742be0df2d809124.jpg](../icml_results/311_Rational%20Transductors/images/160e55651ac5fa0b468081cd378b7503ec0af26c15fb4936742be0df2d809124.jpg)

![208d6e717bd220c1085bcd45687097e0b64a0db4594423325ee25598013aa47c.jpg](../icml_results/311_Rational%20Transductors/images/208d6e717bd220c1085bcd45687097e0b64a0db4594423325ee25598013aa47c.jpg)

![2eb65df01124d63aac79b1fdd4b9652457d06ec62b22bd514e06ac4fc26573a2.jpg](../icml_results/311_Rational%20Transductors/images/2eb65df01124d63aac79b1fdd4b9652457d06ec62b22bd514e06ac4fc26573a2.jpg)

![3e9625dcf696121d5dcde5dc78140fffe3f7b3bcbea25382f924f16335c2bfda.jpg](../icml_results/311_Rational%20Transductors/images/3e9625dcf696121d5dcde5dc78140fffe3f7b3bcbea25382f924f16335c2bfda.jpg)

![482874169186a84627dd9f79219248bf6937dcfd8d0e7617388bdd4a6501c0ec.jpg](../icml_results/311_Rational%20Transductors/images/482874169186a84627dd9f79219248bf6937dcfd8d0e7617388bdd4a6501c0ec.jpg)

![48ce45aa494b95561e34b6fb2b4195427bb0934b6d46bac79e553997dccc5669.jpg](../icml_results/311_Rational%20Transductors/images/48ce45aa494b95561e34b6fb2b4195427bb0934b6d46bac79e553997dccc5669.jpg)

![4ea09a24f979c57c32b36d52a88d00fbadafe897770e41db9938306c19bd6e56.jpg](../icml_results/311_Rational%20Transductors/images/4ea09a24f979c57c32b36d52a88d00fbadafe897770e41db9938306c19bd6e56.jpg)

![760e9102d2512368e4570dd57a3b017f46f96666e97590ad159f375c66b7858b.jpg](../icml_results/311_Rational%20Transductors/images/760e9102d2512368e4570dd57a3b017f46f96666e97590ad159f375c66b7858b.jpg)

![c66617f448acab370ca2cdd24d87e9ce81655e6614ac983b7ab23db33070f6a7.jpg](../icml_results/311_Rational%20Transductors/images/c66617f448acab370ca2cdd24d87e9ce81655e6614ac983b7ab23db33070f6a7.jpg)

### Tables

![0a3c913d072678c0c6c4122087fd57462578ecbd2eef56abe91a1c78524851b0.jpg](../icml_results/311_Rational%20Transductors/tables/0a3c913d072678c0c6c4122087fd57462578ecbd2eef56abe91a1c78524851b0.jpg)

![d9e4a7ec6fd34c8dce76efe55fc988742677424b731261d9c76740760597a549.jpg](../icml_results/311_Rational%20Transductors/tables/d9e4a7ec6fd34c8dce76efe55fc988742677424b731261d9c76740760597a549.jpg)

<a id="paper-0307"></a>

## How much can language models memorize?


### Images

![175401e7d29773087700945af5eeaeda4ba67a2ac1a622e9efcc5ce4f419b15c.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/175401e7d29773087700945af5eeaeda4ba67a2ac1a622e9efcc5ce4f419b15c.jpg)

![3795748f3930ca4b18a25676931c13ced1d15ee378f8cc44c962f04002febab7.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/3795748f3930ca4b18a25676931c13ced1d15ee378f8cc44c962f04002febab7.jpg)

![465d45f13dcd2afc43e515dae11ad5f43ec80fa7b5746a729fd0a4912b6654fb.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/465d45f13dcd2afc43e515dae11ad5f43ec80fa7b5746a729fd0a4912b6654fb.jpg)

![5c37fcdf7a43c0ba771d22e5f324c75dac6e63c31ade3d5e2da4acbb8272e8ca.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/5c37fcdf7a43c0ba771d22e5f324c75dac6e63c31ade3d5e2da4acbb8272e8ca.jpg)

![5e1a23f25fe5ea341ab56dfccddea76f204f30e0bae62d98f38416c646090eab.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/5e1a23f25fe5ea341ab56dfccddea76f204f30e0bae62d98f38416c646090eab.jpg)

![7b1ea044f19e91ed8fd38bd2415e5b91c3fd2ce6650f6597852c5b8cd97f79b3.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/7b1ea044f19e91ed8fd38bd2415e5b91c3fd2ce6650f6597852c5b8cd97f79b3.jpg)

![835528fadc12e8fd20eafc1f61d9213f96ced82b32b4c6228b56f260e5051522.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/835528fadc12e8fd20eafc1f61d9213f96ced82b32b4c6228b56f260e5051522.jpg)

![8d506b7920c8800f18e7a53c12f872b38618d665d8512f119a85d4c029e3a638.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/8d506b7920c8800f18e7a53c12f872b38618d665d8512f119a85d4c029e3a638.jpg)

![a4aa83ad5da5a3a416d075c867366b246e95a2a616ad3a9a49575f6f5d9f69de.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/a4aa83ad5da5a3a416d075c867366b246e95a2a616ad3a9a49575f6f5d9f69de.jpg)

![a4b3f55555d77567f3dabb853828757b35542a50c442b1462585d1693b2ee054.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/a4b3f55555d77567f3dabb853828757b35542a50c442b1462585d1693b2ee054.jpg)

![a9e7d3e4be5d04c623ce5df02fbe6287256abafd86fe7994970cb4f53c247edf.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/a9e7d3e4be5d04c623ce5df02fbe6287256abafd86fe7994970cb4f53c247edf.jpg)

![b199a9db09f54b8378acc16568822c378018d030b63ee3cd0ee5106550e397d6.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/b199a9db09f54b8378acc16568822c378018d030b63ee3cd0ee5106550e397d6.jpg)

![b84f5ccd415dffdddaf8956766f0a4bd09ca0aa755b19305a017e3e92fa69a8c.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/b84f5ccd415dffdddaf8956766f0a4bd09ca0aa755b19305a017e3e92fa69a8c.jpg)

![bb895d0f005758028487bb0e9eae78829c1a6f8689478fcc0be5e5066bfdabbc.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/bb895d0f005758028487bb0e9eae78829c1a6f8689478fcc0be5e5066bfdabbc.jpg)

![bda8b9949bb80a329996872a5492eb006cc7dcb3a6ed7906b03dec950fef705c.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/bda8b9949bb80a329996872a5492eb006cc7dcb3a6ed7906b03dec950fef705c.jpg)

![e1dc4abf6793145a54e47d20f3f23630b7d1a715ab867ab517227a1cd4179734.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/e1dc4abf6793145a54e47d20f3f23630b7d1a715ab867ab517227a1cd4179734.jpg)

![f88a6c2008753003633fcd81b6c95039adb2989223c349f940dda1c5bee0592f.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/f88a6c2008753003633fcd81b6c95039adb2989223c349f940dda1c5bee0592f.jpg)

![ff297f2f998811d38e36e0df413ea35cec97effe36f4c0b3f9a3717f85d750ed.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/images/ff297f2f998811d38e36e0df413ea35cec97effe36f4c0b3f9a3717f85d750ed.jpg)

### Tables

![28a4435d5e9e44309146b29e31f7e9599cfc9f5a538c507bdf383094d3c2f465.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/tables/28a4435d5e9e44309146b29e31f7e9599cfc9f5a538c507bdf383094d3c2f465.jpg)

![b8cc6115990214444d1347040810ee34a0c9b103f4e27d1680e18001bdef056c.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/tables/b8cc6115990214444d1347040810ee34a0c9b103f4e27d1680e18001bdef056c.jpg)

![c195bb1860eb8cf7423098308129a595ada81f03da754efa993e690a715ad880.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/tables/c195bb1860eb8cf7423098308129a595ada81f03da754efa993e690a715ad880.jpg)

![d1f43894717bcf070e30a049769c926ab359e29c8af04107d81dddfef0a77195.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/tables/d1f43894717bcf070e30a049769c926ab359e29c8af04107d81dddfef0a77195.jpg)

![d8dfd149392498a0640b699c14c07efa4b383f7e87158cd31d97322bf1028ab8.jpg](../icml_results/312_How%20much%20can%20language%20models%20memorize_/tables/d8dfd149392498a0640b699c14c07efa4b383f7e87158cd31d97322bf1028ab8.jpg)

<a id="paper-0308"></a>

## Post-Training with Policy Gradients: Optimality and the Base Model Barrier


### Images

![027ca5a0784184160c38db7ce7d53f00f3302adcd26a904cd0cdb147cd2f2f4e.jpg](../icml_results/313_Post-Training%20with%20Policy%20Gradients_%20Optimality%20and%20the%20Base%20Model%20Barrier/images/027ca5a0784184160c38db7ce7d53f00f3302adcd26a904cd0cdb147cd2f2f4e.jpg)

![284e6f3b9996627ad22e3a651953b5de94a5e597060ab40bc41ac7f85cbc224d.jpg](../icml_results/313_Post-Training%20with%20Policy%20Gradients_%20Optimality%20and%20the%20Base%20Model%20Barrier/images/284e6f3b9996627ad22e3a651953b5de94a5e597060ab40bc41ac7f85cbc224d.jpg)

### Tables

![8d347abb02a83176e9fd5b9a1061897342e2c798656c1498e1caeac72e3af957.jpg](../icml_results/313_Post-Training%20with%20Policy%20Gradients_%20Optimality%20and%20the%20Base%20Model%20Barrier/tables/8d347abb02a83176e9fd5b9a1061897342e2c798656c1498e1caeac72e3af957.jpg)

<a id="paper-0309"></a>

## On the Convergence Rate of LoRA Gradient Descent


### Images

![1c0dc8b15a5b757ef4a99fdb438a5c53e93c1afd327feecba20e80a37fd36941.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/1c0dc8b15a5b757ef4a99fdb438a5c53e93c1afd327feecba20e80a37fd36941.jpg)

![2d8de4e81ad455e0d0bbb0156544316a1e9a931f4966611cf133085fcf80732a.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/2d8de4e81ad455e0d0bbb0156544316a1e9a931f4966611cf133085fcf80732a.jpg)

![3c0ea6045f5ca4b6b719afda75e955d968cdc42d6ebe1104e6bbdb64f03f6494.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/3c0ea6045f5ca4b6b719afda75e955d968cdc42d6ebe1104e6bbdb64f03f6494.jpg)

![970503a66118dbf93440cf9fa132ae754dde1784825613ab828b02dba4b2554d.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/970503a66118dbf93440cf9fa132ae754dde1784825613ab828b02dba4b2554d.jpg)

![cd58a51049c015fcacd75f280458ade3b1bba0c472806055ded37757ddf15dfc.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/cd58a51049c015fcacd75f280458ade3b1bba0c472806055ded37757ddf15dfc.jpg)

![dd94c17f7b9293fa4a085d8730547e46e4832903f9cc8853421a61dda09307fe.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/dd94c17f7b9293fa4a085d8730547e46e4832903f9cc8853421a61dda09307fe.jpg)

![e97508752d00883417c3584183fad008dfab7081664f694377f9c95d42fe12f6.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/e97508752d00883417c3584183fad008dfab7081664f694377f9c95d42fe12f6.jpg)

![ff674339639d5982855b32f132aa99e7a9d085649f3d0faba1cfac82c3001853.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/images/ff674339639d5982855b32f132aa99e7a9d085649f3d0faba1cfac82c3001853.jpg)

### Tables

![4a0987d619d17fb4d6be866d2312688d25ebf48768b712de9005ce26e0a6853e.jpg](../icml_results/314_On%20the%20Convergence%20Rate%20of%20LoRA%20Gradient%20Descent/tables/4a0987d619d17fb4d6be866d2312688d25ebf48768b712de9005ce26e0a6853e.jpg)

<a id="paper-0310"></a>

## Do We Need Adam? Surprisingly Strong and Sparse Reinforcement Learning with SGD in LLMs


### Images

![18a7cfa866d7a4a05a7d793d5f07508949932baa2384cf9cbad0340c8d5d2116.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/images/18a7cfa866d7a4a05a7d793d5f07508949932baa2384cf9cbad0340c8d5d2116.jpg)

![235d734b137841380a026517d76396bc22c0f9fafa64eda708c6aa0c1d97bf8a.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/images/235d734b137841380a026517d76396bc22c0f9fafa64eda708c6aa0c1d97bf8a.jpg)

![247c28880d24245399421565f452411a6b5a233f3156a2b7d0f13a5528d473c1.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/images/247c28880d24245399421565f452411a6b5a233f3156a2b7d0f13a5528d473c1.jpg)

![a930bef0a32bb39b82ddc60de833ac646c576a691d94e4cee1d218a2b80b4151.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/images/a930bef0a32bb39b82ddc60de833ac646c576a691d94e4cee1d218a2b80b4151.jpg)

![b9e09df5b1720b7da6aef19210fa2ad71bd3e8c37dbc8951c67bab87cba4334d.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/images/b9e09df5b1720b7da6aef19210fa2ad71bd3e8c37dbc8951c67bab87cba4334d.jpg)

![c5dcbe0d45bc18ecec2b9ef95f31268cb3cb2e1237922ea615331227be748367.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/images/c5dcbe0d45bc18ecec2b9ef95f31268cb3cb2e1237922ea615331227be748367.jpg)

### Tables

![4693715fce6335412e432f0f285f56f81f6035d03f1fe162df832b41fed3bd60.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/4693715fce6335412e432f0f285f56f81f6035d03f1fe162df832b41fed3bd60.jpg)

![796b0181e6a82c92d923655f9a582dc97fe4c3b44c8dd0d6a04a96c52d2ae219.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/796b0181e6a82c92d923655f9a582dc97fe4c3b44c8dd0d6a04a96c52d2ae219.jpg)

![96e947b38fda72e3414d82d4315b037fe8a8bca52f6392f8ef72f81f845875ee.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/96e947b38fda72e3414d82d4315b037fe8a8bca52f6392f8ef72f81f845875ee.jpg)

![9e911cd52457c6b956fba7c5aa111c09fff65af3b9a4f639914620e4ee6af3b9.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/9e911cd52457c6b956fba7c5aa111c09fff65af3b9a4f639914620e4ee6af3b9.jpg)

![b3169c1e789223ba14fe9ed20e795b9f5fc3b23b09557a1d7ad03937b3484a20.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/b3169c1e789223ba14fe9ed20e795b9f5fc3b23b09557a1d7ad03937b3484a20.jpg)

![e362c8334ce616828d6dd14b3c4ac30210f60f76d036fe491a8514b7e5e86ab7.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/e362c8334ce616828d6dd14b3c4ac30210f60f76d036fe491a8514b7e5e86ab7.jpg)

![fc74ccd7c649196fa15129e09507a5dbf0f145505f3e44b9490e4718ba534366.jpg](../icml_results/315_Do%20We%20Need%20Adam_%20Surprisingly%20Strong%20and%20Sparse%20Reinforcement%20Learning%20with%20SGD%20in%20LLMs/tables/fc74ccd7c649196fa15129e09507a5dbf0f145505f3e44b9490e4718ba534366.jpg)

<a id="paper-0311"></a>

## Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity


### Images

![76297a47ed84132fa5c39966b75342cf088f13986326b1b000242169ea41ca47.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/images/76297a47ed84132fa5c39966b75342cf088f13986326b1b000242169ea41ca47.jpg)

![8490cf9958bb55c8cd21b73104ef262128a505585c174429f13eb76e33a54048.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/images/8490cf9958bb55c8cd21b73104ef262128a505585c174429f13eb76e33a54048.jpg)

![a7915ea640c983c45c938c1345b49f3f69a510c1664c42052be6c7ffb0003efb.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/images/a7915ea640c983c45c938c1345b49f3f69a510c1664c42052be6c7ffb0003efb.jpg)

![b85d7dbf004a198a0d82f9538eb3bca4b4baa802e92df5cda6fa8b7363b41741.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/images/b85d7dbf004a198a0d82f9538eb3bca4b4baa802e92df5cda6fa8b7363b41741.jpg)

![fdac83308343e1c11614807a8890f4e64c51df0c13ae3c81426382b5d09081df.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/images/fdac83308343e1c11614807a8890f4e64c51df0c13ae3c81426382b5d09081df.jpg)

![fdb72116b82f592efa720ec2e82d2a9d229851e78c153a69bfcc2b8a71e45ca2.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/images/fdb72116b82f592efa720ec2e82d2a9d229851e78c153a69bfcc2b8a71e45ca2.jpg)

### Tables

![63ba1d5b587d85d32122c1d49b545a1f76fd77291cc51d7c66ffafaa75edb0a6.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/tables/63ba1d5b587d85d32122c1d49b545a1f76fd77291cc51d7c66ffafaa75edb0a6.jpg)

![f03f8855c447767397dba5e61b42b97bd6d8b73912a24441e44f10d575f6a334.jpg](../icml_results/316_Reward%20Redistribution%20for%20CVaR%20MDPs%20using%20a%20Bellman%20Operator%20on%20L-infinity/tables/f03f8855c447767397dba5e61b42b97bd6d8b73912a24441e44f10d575f6a334.jpg)

<a id="paper-0312"></a>

## Flowers: A Warp Drive for Neural PDE Solvers


### Images

![0ffb5ea60adaa25dc77e8cd23306b7590602d1e9a9719450acbb6e5ac376e343.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/0ffb5ea60adaa25dc77e8cd23306b7590602d1e9a9719450acbb6e5ac376e343.jpg)

![1071b2f1d1629c5edd5725801360b43d46f04a96617bf00c3f660cf1794e651d.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/1071b2f1d1629c5edd5725801360b43d46f04a96617bf00c3f660cf1794e651d.jpg)

![132ac564540725b36394ca4351f813574f1c645d6559876119e387d1aaa69757.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/132ac564540725b36394ca4351f813574f1c645d6559876119e387d1aaa69757.jpg)

![50965d794ffc2361b0746dcdd4a558e87090ec79b1177bcaf70151aa84e6e9f3.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/50965d794ffc2361b0746dcdd4a558e87090ec79b1177bcaf70151aa84e6e9f3.jpg)

![75f76e8598e427acce7638366e9807a06d2528042313c70c615838b4e84fb492.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/75f76e8598e427acce7638366e9807a06d2528042313c70c615838b4e84fb492.jpg)

![7ce33fab72634207737432b553f5dcbce0f59ac86cb7a467fff2c602a45b1a39.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/7ce33fab72634207737432b553f5dcbce0f59ac86cb7a467fff2c602a45b1a39.jpg)

![89aaafbcfae6bdcb1a1ee0d2d5d34c9c16ebbb21560fc57280515a2475a6a22f.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/89aaafbcfae6bdcb1a1ee0d2d5d34c9c16ebbb21560fc57280515a2475a6a22f.jpg)

![94203c27ccb165e86dd698fa5f2361cd0f446ed3a7d8f25213024dd4aae70083.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/94203c27ccb165e86dd698fa5f2361cd0f446ed3a7d8f25213024dd4aae70083.jpg)

![b04945031cf06774ffd5d179eaca2ae9291feb51f0a7c67870fc2cc0d0f1b2b4.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/b04945031cf06774ffd5d179eaca2ae9291feb51f0a7c67870fc2cc0d0f1b2b4.jpg)

![b9d355b8b99ccc40e1eca43479194af86ceb58dc4413f2fa26a117d7807471a6.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/b9d355b8b99ccc40e1eca43479194af86ceb58dc4413f2fa26a117d7807471a6.jpg)

![d731bdb4b1c4dd5905ffa11a49da5927485e435979e2289703689a4c212d20f7.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/d731bdb4b1c4dd5905ffa11a49da5927485e435979e2289703689a4c212d20f7.jpg)

![df69d111c316a20bd9ea14a7088399bf4d55b8bba8f6550c078baf4df4838e54.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/df69d111c316a20bd9ea14a7088399bf4d55b8bba8f6550c078baf4df4838e54.jpg)

![ec422fb1e15c2f4a839f6c5d6666dc89a9545afb73ec2eb16bab1c47fb190eee.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/ec422fb1e15c2f4a839f6c5d6666dc89a9545afb73ec2eb16bab1c47fb190eee.jpg)

![f10d18addcbe80a5506697005fb78c5e1edfda5a7c352ed7b88fc6931ce0a333.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/images/f10d18addcbe80a5506697005fb78c5e1edfda5a7c352ed7b88fc6931ce0a333.jpg)

### Tables

![136e9fe2bd7f84e63f8dbbfb6fd6fd59829f8600ba1a7851e6f20e69d63b958d.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/136e9fe2bd7f84e63f8dbbfb6fd6fd59829f8600ba1a7851e6f20e69d63b958d.jpg)

![3f5316841a79ea89e753ccc60a4cb70ea3c822f43dba08a36ccf6f6c01c164fd.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/3f5316841a79ea89e753ccc60a4cb70ea3c822f43dba08a36ccf6f6c01c164fd.jpg)

![61e1d269559558675ad6b8c5002f588d746e63a7daec1e615a0fe6205179a7a0.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/61e1d269559558675ad6b8c5002f588d746e63a7daec1e615a0fe6205179a7a0.jpg)

![68fa078f3066427cfd6bd1a1b5123d9761f8ff536215d7e56028b865efcadaed.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/68fa078f3066427cfd6bd1a1b5123d9761f8ff536215d7e56028b865efcadaed.jpg)

![86ffea5e5dbfff1c29cf072f5f785c7e97030fe42404bf8531d49d51fb3daad1.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/86ffea5e5dbfff1c29cf072f5f785c7e97030fe42404bf8531d49d51fb3daad1.jpg)

![b81c609e360fe7107279041f13ec3a7be334b0271565f96feba868281b1a17d7.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/b81c609e360fe7107279041f13ec3a7be334b0271565f96feba868281b1a17d7.jpg)

![bf8c2e6d79a5b4ae038580ee87dba6ec0285043997eabd1b5964ca437bf6b571.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/bf8c2e6d79a5b4ae038580ee87dba6ec0285043997eabd1b5964ca437bf6b571.jpg)

![cc777adff4317a13051483cc18546709d4e2e72b19906f32755047ff2d4c4ecd.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/cc777adff4317a13051483cc18546709d4e2e72b19906f32755047ff2d4c4ecd.jpg)

![da085282000679fc978a72f008c117e996c4088d91aa7847997082d745af9fa4.jpg](../icml_results/317_Flowers_%20A%20Warp%20Drive%20for%20Neural%20PDE%20Solvers/tables/da085282000679fc978a72f008c117e996c4088d91aa7847997082d745af9fa4.jpg)

<a id="paper-0313"></a>

## Accelerating Q-learning through Efficient Value-Sharing across Actions


### Images

![06462024243bd202bdfbe0aa24c323d0077647df40bc11ec71b70d05020846f8.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/06462024243bd202bdfbe0aa24c323d0077647df40bc11ec71b70d05020846f8.jpg)

![0a7767b2495ddda4b970f159bfce7563d1c912da9ad5a3756d2b59130a89de99.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/0a7767b2495ddda4b970f159bfce7563d1c912da9ad5a3756d2b59130a89de99.jpg)

![545fb64773efaa4f820ae4e638839658e03319e340504af72d57cd9e813d12cd.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/545fb64773efaa4f820ae4e638839658e03319e340504af72d57cd9e813d12cd.jpg)

![65b6849b0031efb484dc28c3f9423a2ce44679d03b5e5eaa5a957e10c674bfa1.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/65b6849b0031efb484dc28c3f9423a2ce44679d03b5e5eaa5a957e10c674bfa1.jpg)

![74093f133ae812105c59d2f67b91b7db410246ec4145ab3dca558b5ac2d36d83.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/74093f133ae812105c59d2f67b91b7db410246ec4145ab3dca558b5ac2d36d83.jpg)

![827cedc231f150ed319f3bf6842e0d37a93ce684d9d3943f9e2489d3bff25612.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/827cedc231f150ed319f3bf6842e0d37a93ce684d9d3943f9e2489d3bff25612.jpg)

![883da9d11f2f994d73be2776c5292ee0e438e305d000fbfe5517f1935e02085c.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/883da9d11f2f994d73be2776c5292ee0e438e305d000fbfe5517f1935e02085c.jpg)

![af02e72097ca691534c6a68300ad636acde390bda2b7054a6d8a3c28d7305058.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/af02e72097ca691534c6a68300ad636acde390bda2b7054a6d8a3c28d7305058.jpg)

![ce09b498829dcaffd4eb9ef396709ca19543f911e0e2dc1332dd3531ee48e129.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/ce09b498829dcaffd4eb9ef396709ca19543f911e0e2dc1332dd3531ee48e129.jpg)

![e0d7940c6b972b2569ab86633b044a092b49ffdf3ccd60b10be4f208f8a1be24.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/images/e0d7940c6b972b2569ab86633b044a092b49ffdf3ccd60b10be4f208f8a1be24.jpg)

### Tables

![115778d13e334d673299717f27c89331b66e762cc5eb6588ce875a3bdfb87d97.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/tables/115778d13e334d673299717f27c89331b66e762cc5eb6588ce875a3bdfb87d97.jpg)

![577bc11581e989fd13ee979d6ec9f3b3588903927e77f28a02b507d55d37cb8c.jpg](../icml_results/318_Accelerating%20Q-learning%20through%20Efficient%20Value-Sharing%20across%20Actions/tables/577bc11581e989fd13ee979d6ec9f3b3588903927e77f28a02b507d55d37cb8c.jpg)

<a id="paper-0314"></a>

## HDFlow: Hierarchical Diffusion-Flow Planning for Long-horizon Tasks


### Images

![0d8a5d1e613bb48b0809b8da2a1251151493c5a4912698572ba78a4128c16967.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/0d8a5d1e613bb48b0809b8da2a1251151493c5a4912698572ba78a4128c16967.jpg)

![3b4063d4278edcae0d68ba423d2a0169f14a5cb0bfba55c7f52cfbbc1ed0fbec.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/3b4063d4278edcae0d68ba423d2a0169f14a5cb0bfba55c7f52cfbbc1ed0fbec.jpg)

![4d99a1033af97b637383bd2a78b17ff16ee83d0ed996642e6f3b8c81e5b7df25.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/4d99a1033af97b637383bd2a78b17ff16ee83d0ed996642e6f3b8c81e5b7df25.jpg)

![5de5c53639101f08aaafcaa4ff0fc0684898ba5bc73df4a1adc57ffb23fff7ae.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/5de5c53639101f08aaafcaa4ff0fc0684898ba5bc73df4a1adc57ffb23fff7ae.jpg)

![70795d06595217fbf1b122681cb147f9150427fb227aef5bd2dfbd06514b2605.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/70795d06595217fbf1b122681cb147f9150427fb227aef5bd2dfbd06514b2605.jpg)

![8200a415a183ab8502ad0146362b3cafd9ef9976c90db956742476692a30e749.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/8200a415a183ab8502ad0146362b3cafd9ef9976c90db956742476692a30e749.jpg)

![893f8bd7d06b5779bd8550ba2cf5e4dca26350a0434e3ae2b14e5db6411696ce.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/893f8bd7d06b5779bd8550ba2cf5e4dca26350a0434e3ae2b14e5db6411696ce.jpg)

![8fda1e5a72d20addb193432ed93b5fdd3bcba6273ee836b9fe2f2da3fcaea865.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/8fda1e5a72d20addb193432ed93b5fdd3bcba6273ee836b9fe2f2da3fcaea865.jpg)

![9ac762b08827d8ec9bf4cdf68f4ea6ea2240086dfb8aa6c72b25611e65fd8328.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/9ac762b08827d8ec9bf4cdf68f4ea6ea2240086dfb8aa6c72b25611e65fd8328.jpg)

![ced9969425800c3a636904e35b7c6cfdee15d1e5324a8d7d4c80c0bee0146059.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/ced9969425800c3a636904e35b7c6cfdee15d1e5324a8d7d4c80c0bee0146059.jpg)

![d6b85efc4680b1cac4ff4cb4b3107852bf08f507b6f1dbc0ba5163d1e27d7375.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/d6b85efc4680b1cac4ff4cb4b3107852bf08f507b6f1dbc0ba5163d1e27d7375.jpg)

![e062a45bf5a0f2c1b3b808b71fb59030035183ad0e6ea308852533774c4d104f.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/e062a45bf5a0f2c1b3b808b71fb59030035183ad0e6ea308852533774c4d104f.jpg)

![e11f98bf99ad44aaccd780bca9bbe83a42a7acdddc0080f46349b924e9daf377.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/e11f98bf99ad44aaccd780bca9bbe83a42a7acdddc0080f46349b924e9daf377.jpg)

![ebd2351009b0fcba4b169af33a9927ee7429f0b12cdb9ff83ac886ecd8719b25.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/ebd2351009b0fcba4b169af33a9927ee7429f0b12cdb9ff83ac886ecd8719b25.jpg)

![f54c89b46d1a145abf3dd86488a98fd8675a806befc495441ac86f3dc74b352a.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/f54c89b46d1a145abf3dd86488a98fd8675a806befc495441ac86f3dc74b352a.jpg)

![ff2664bb5d9fa07717f070c0e13ee7dde6de65925bbe05eae611eee4096d53bf.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/images/ff2664bb5d9fa07717f070c0e13ee7dde6de65925bbe05eae611eee4096d53bf.jpg)

### Tables

![0b525062a1e010aaa371f31b4b08a0219cbb175a2d91f4d7b2c1515ee25885c1.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/0b525062a1e010aaa371f31b4b08a0219cbb175a2d91f4d7b2c1515ee25885c1.jpg)

![18c58372ecb00b2bfc0693118be6f31be7b50c69254bee133b67a49a07a51fd3.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/18c58372ecb00b2bfc0693118be6f31be7b50c69254bee133b67a49a07a51fd3.jpg)

![1e28758367e27dc732d1927a8119b8f9e368b03564566e6c67f0efac1c2aa389.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/1e28758367e27dc732d1927a8119b8f9e368b03564566e6c67f0efac1c2aa389.jpg)

![39fa310d51000c144c8eb02d1ccde6d7b2a3548e3f66af076825fdb0b0105b1d.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/39fa310d51000c144c8eb02d1ccde6d7b2a3548e3f66af076825fdb0b0105b1d.jpg)

![4195ea230dae706c6972a73a80e77608aaee1187669b6854c17708026fcdd529.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/4195ea230dae706c6972a73a80e77608aaee1187669b6854c17708026fcdd529.jpg)

![599770fb3f955cbd27ab3e5c757be9d15f4baa8f007bccd147b3adfb5862054d.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/599770fb3f955cbd27ab3e5c757be9d15f4baa8f007bccd147b3adfb5862054d.jpg)

![64a2fde882d32cbc76dd08bd211fe840feb431021c95d24d9c41e3f6a072c9b8.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/64a2fde882d32cbc76dd08bd211fe840feb431021c95d24d9c41e3f6a072c9b8.jpg)

![69a610652336c70870701df4e0f899a46596d1abf4afcc52b7e5f4ebd08d8038.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/69a610652336c70870701df4e0f899a46596d1abf4afcc52b7e5f4ebd08d8038.jpg)

![6d7196307f3c858649fbd9ae97389bcf624da4491c95cb96bafc89b15e57de80.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/6d7196307f3c858649fbd9ae97389bcf624da4491c95cb96bafc89b15e57de80.jpg)

![7d18c5632d2ad59cec71b87417a2bbde689513c814f0c6df9d188d449f7f599e.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/7d18c5632d2ad59cec71b87417a2bbde689513c814f0c6df9d188d449f7f599e.jpg)

![a3b84803acf44a27b7fed6845fb11e65b7d7d140354a9fc2ac7aa717803d5489.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/a3b84803acf44a27b7fed6845fb11e65b7d7d140354a9fc2ac7aa717803d5489.jpg)

![ac7fef33faf322b2a272a7d60b6a5a7f48c844027c2d92c0b5d0338bdc48858c.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/ac7fef33faf322b2a272a7d60b6a5a7f48c844027c2d92c0b5d0338bdc48858c.jpg)

![db368d4831285b9e6f9c0c61cd89c137b1294d140b16e75652e8b61e01185ab8.jpg](../icml_results/319_HDFlow_%20Hierarchical%20Diffusion-Flow%20Planning%20for%20Long-horizon%20Tasks/tables/db368d4831285b9e6f9c0c61cd89c137b1294d140b16e75652e8b61e01185ab8.jpg)

<a id="paper-0315"></a>

## MuonSSM: Orthogonalizing State Space Models for Sequence Modeling


### Images

![009d35a334a5ca56fd3aa21686e962e22696ca7f7e442aabe086e4eb6f7ad044.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/009d35a334a5ca56fd3aa21686e962e22696ca7f7e442aabe086e4eb6f7ad044.jpg)

![0f5a36478c66acc85d50ad67b60a3a272463b47fa2f9332047561b236efccc6e.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/0f5a36478c66acc85d50ad67b60a3a272463b47fa2f9332047561b236efccc6e.jpg)

![461cd46150a29daaee3be4ca939e324721846e52feed95e213b98cd414438080.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/461cd46150a29daaee3be4ca939e324721846e52feed95e213b98cd414438080.jpg)

![669592042193506af06b0d3ad3880247d0d7313246faa1c8777dc88e59989ba0.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/669592042193506af06b0d3ad3880247d0d7313246faa1c8777dc88e59989ba0.jpg)

![6f02734a1e3f0791e89f2450e21fdc4d005d02792c65344a5a5428d636ba5027.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/6f02734a1e3f0791e89f2450e21fdc4d005d02792c65344a5a5428d636ba5027.jpg)

![c208f4774fa0b1f7817da193f99fcabe2aa5d98a04f27e27ca7bef4950f98023.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/c208f4774fa0b1f7817da193f99fcabe2aa5d98a04f27e27ca7bef4950f98023.jpg)

![ea2b216cfd040c7a5e911f5bdae6e935e91bffdf3298046c891d67a7618bb089.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/images/ea2b216cfd040c7a5e911f5bdae6e935e91bffdf3298046c891d67a7618bb089.jpg)

### Tables

![0708de77638f500e6a9388ac3a70ae156be779c1330111773703350593700cbd.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/0708de77638f500e6a9388ac3a70ae156be779c1330111773703350593700cbd.jpg)

![12e6b1a12a03ad5b252c9d8a1ebd9cbc30e8bf4d05a92781b95c5014b1fba0dd.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/12e6b1a12a03ad5b252c9d8a1ebd9cbc30e8bf4d05a92781b95c5014b1fba0dd.jpg)

![1442b0de98729ae4f2828961a5ed3bb7384bb459682b239dd7239cbdeb5e9ad2.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/1442b0de98729ae4f2828961a5ed3bb7384bb459682b239dd7239cbdeb5e9ad2.jpg)

![181a3d1f5cf2cb47fc3be07315c73a542264ee38452812c2f8b75ecaea4c2514.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/181a3d1f5cf2cb47fc3be07315c73a542264ee38452812c2f8b75ecaea4c2514.jpg)

![347debbfd7f3b480495359ede9717cc23a72a6345eadf646250c220cd6e2c1b6.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/347debbfd7f3b480495359ede9717cc23a72a6345eadf646250c220cd6e2c1b6.jpg)

![409331d5814d562c901f239e93e55f242223dd2345beea81b91290a512de306a.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/409331d5814d562c901f239e93e55f242223dd2345beea81b91290a512de306a.jpg)

![43d0508a0fe7af16683d08c914133737c1ced368e3dd44c131b15a63390908a1.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/43d0508a0fe7af16683d08c914133737c1ced368e3dd44c131b15a63390908a1.jpg)

![584ed2a3c851e296df626915e53bcbf68fc130ed32d24c6276986ef14c747b86.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/584ed2a3c851e296df626915e53bcbf68fc130ed32d24c6276986ef14c747b86.jpg)

![76321ad6d3a28d582984fb3a0dbbbd46846a2b27ba5519667b3cb75faf68d783.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/76321ad6d3a28d582984fb3a0dbbbd46846a2b27ba5519667b3cb75faf68d783.jpg)

![8235b1612450b214b8c74eea4bce3acad9d2c7bb805a8fb75113e903d6c691e8.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/8235b1612450b214b8c74eea4bce3acad9d2c7bb805a8fb75113e903d6c691e8.jpg)

![9850c4c8261a88d0be027941f67fff9ec1d0d6dbe43a781e0cfca98b513121a8.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/9850c4c8261a88d0be027941f67fff9ec1d0d6dbe43a781e0cfca98b513121a8.jpg)

![9c794538416f9915f0523726dd880579238ea2d5ff6e468b9e6c4839feedcaaf.jpg](../icml_results/320_MuonSSM_%20Orthogonalizing%20State%20Space%20Models%20for%20Sequence%20Modeling/tables/9c794538416f9915f0523726dd880579238ea2d5ff6e468b9e6c4839feedcaaf.jpg)

<a id="paper-0316"></a>

## Exact Unlearning in Reinforcement Learning


### Images

![01ad6b8774d60e4087ed3b8c88e92e18477ff91c437ee7b43e6b47501770785b.jpg](../icml_results/321_Exact%20Unlearning%20in%20Reinforcement%20Learning/images/01ad6b8774d60e4087ed3b8c88e92e18477ff91c437ee7b43e6b47501770785b.jpg)

<a id="paper-0317"></a>

## The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models


### Images

![1f9e76af20808139f6a2d87a6e45b5a730e6fad1e9722921363561f09fe0f0ad.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/1f9e76af20808139f6a2d87a6e45b5a730e6fad1e9722921363561f09fe0f0ad.jpg)

![2ba1d0735f398e658699ae4dbffb63d9af554c3ce6a0670dac0a68a3aa657581.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/2ba1d0735f398e658699ae4dbffb63d9af554c3ce6a0670dac0a68a3aa657581.jpg)

![53f01f7aa80da48f152a5fc8a7d05c329271fadb79dd171cdf61b9c764a359ba.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/53f01f7aa80da48f152a5fc8a7d05c329271fadb79dd171cdf61b9c764a359ba.jpg)

![8220606c4b7620a72f1e62b33b087f0bcacec6fbdb4e9cbb5adad1ae35a58fda.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/8220606c4b7620a72f1e62b33b087f0bcacec6fbdb4e9cbb5adad1ae35a58fda.jpg)

![88fa29f74c61f994ba489bd309846653fa5956d6d75f539b9e5f207cb056a269.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/88fa29f74c61f994ba489bd309846653fa5956d6d75f539b9e5f207cb056a269.jpg)

![94c0e5ce09cef6478e2152839fe990df20c2cfe4894c214d39625707d153257c.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/94c0e5ce09cef6478e2152839fe990df20c2cfe4894c214d39625707d153257c.jpg)

![98813867ed3cfc12604c0f4e47ec935488b58ad373d1c74126356232123d45ab.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/98813867ed3cfc12604c0f4e47ec935488b58ad373d1c74126356232123d45ab.jpg)

![98afdaa52adb831231d93d8d8f1944df52dcd025e4c8cb6407242604558d7de8.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/98afdaa52adb831231d93d8d8f1944df52dcd025e4c8cb6407242604558d7de8.jpg)

![a3a6d499b6e9fd70ce206c904e4a84315342c44e4707504df7d1f227be786203.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/a3a6d499b6e9fd70ce206c904e4a84315342c44e4707504df7d1f227be786203.jpg)

![a74a1b819b0b47e0b6de72de49c9801dc3ddb8f39fe769cd7ac4ad8a9d4ac1c4.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/a74a1b819b0b47e0b6de72de49c9801dc3ddb8f39fe769cd7ac4ad8a9d4ac1c4.jpg)

![af195e42c2f723eda9602ef08347708d180630445e77423a8729b7fc232d5c3d.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/af195e42c2f723eda9602ef08347708d180630445e77423a8729b7fc232d5c3d.jpg)

![bc824814357f6a1bc3fe257911e13bef7c61e482ed99f23c0881c260142fd7ed.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/images/bc824814357f6a1bc3fe257911e13bef7c61e482ed99f23c0881c260142fd7ed.jpg)

### Tables

![007cffd30a4278bc15362d4a0afd0da348290771db7b0e178323c83d43c94a51.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/tables/007cffd30a4278bc15362d4a0afd0da348290771db7b0e178323c83d43c94a51.jpg)

![0690d62e50100650d5ea969d2c7c18ac8236f1fef379408062245ce6c69163c9.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/tables/0690d62e50100650d5ea969d2c7c18ac8236f1fef379408062245ce6c69163c9.jpg)

![2d2d90bbd994051375521ae50bee1815bfec4147d45862a481916493b162f6b2.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/tables/2d2d90bbd994051375521ae50bee1815bfec4147d45862a481916493b162f6b2.jpg)

![8401976a617f9e0fc900d39172b6b9a88c3e2552afd53fa5828cf88f169632ba.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/tables/8401976a617f9e0fc900d39172b6b9a88c3e2552afd53fa5828cf88f169632ba.jpg)

![8a486122d19c51dae4b22508be1f3a8e9dce7bd8a6028c48fd5cab7716b0261b.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/tables/8a486122d19c51dae4b22508be1f3a8e9dce7bd8a6028c48fd5cab7716b0261b.jpg)

![f32b3a629f48bdb68dd12080eb8611e7fa94009585ff9f022e0930f4dc173250.jpg](../icml_results/322_The%20Flexibility%20Trap_%20Rethinking%20the%20Value%20of%20Arbitrary%20Order%20in%20Diffusion%20Language%20Models/tables/f32b3a629f48bdb68dd12080eb8611e7fa94009585ff9f022e0930f4dc173250.jpg)

<a id="paper-0318"></a>

## Loss-Aware Distributionally Robust Optimization via Trainable Optimal Transport Ambiguity Sets


### Images

![0cca8d5480f833cef58cc3de8c139d46b11177b7dd5d2a168ea680f94ee826d2.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/0cca8d5480f833cef58cc3de8c139d46b11177b7dd5d2a168ea680f94ee826d2.jpg)

![167fc07bd233e6e3e2a6c5607004da043c6165aeeb85622a3e758784352f4c96.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/167fc07bd233e6e3e2a6c5607004da043c6165aeeb85622a3e758784352f4c96.jpg)

![356255d7784a062329f263f57da727b7c69190753b8fb5838ff02d9b64f5bb21.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/356255d7784a062329f263f57da727b7c69190753b8fb5838ff02d9b64f5bb21.jpg)

![412b081b0e34a5076da4d3dbf720b70f2840744e637e50203fc5158ff996ff83.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/412b081b0e34a5076da4d3dbf720b70f2840744e637e50203fc5158ff996ff83.jpg)

![41340e104551678c6e6800666fad05354f7949052b97640f2dc6f8f87ea98552.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/41340e104551678c6e6800666fad05354f7949052b97640f2dc6f8f87ea98552.jpg)

![42f9b00654767ffb84217190cee4919b097b89af0128a2bd39e37d57d619722d.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/42f9b00654767ffb84217190cee4919b097b89af0128a2bd39e37d57d619722d.jpg)

![6f3e41370f8604538c4cda11ea889d788d0062d0f17ee1733c7fab8e79cbdadd.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/6f3e41370f8604538c4cda11ea889d788d0062d0f17ee1733c7fab8e79cbdadd.jpg)

![8039cea0b267a6f71f2982abc81dd9b0f31dc516ba23acee61adf07166a7c020.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/8039cea0b267a6f71f2982abc81dd9b0f31dc516ba23acee61adf07166a7c020.jpg)

![878ab4ef47460557561f0665ec50b4ad6c39864478c766a1b6cc088c1bd06160.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/878ab4ef47460557561f0665ec50b4ad6c39864478c766a1b6cc088c1bd06160.jpg)

![88d18353a42da604e6b690d4afddba4c298d1e3a6f193d9928fe8265573dc31b.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/88d18353a42da604e6b690d4afddba4c298d1e3a6f193d9928fe8265573dc31b.jpg)

![a17074b4324a64aa22381d388832c6215b46417aa8f03ed7394120eede4b3af6.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/a17074b4324a64aa22381d388832c6215b46417aa8f03ed7394120eede4b3af6.jpg)

![b01ca6f8c1aa0dcbe3230c20b73eeff04f445bb5bea9c4240ef4233ee7425c0e.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/b01ca6f8c1aa0dcbe3230c20b73eeff04f445bb5bea9c4240ef4233ee7425c0e.jpg)

![b247b8a0bdb8e518a1e0b66f819e848c0df076a0bb0c4e8e9e8710ffdbd91ab5.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/b247b8a0bdb8e518a1e0b66f819e848c0df076a0bb0c4e8e9e8710ffdbd91ab5.jpg)

![b709f18fbd968e94f1a81c674a917cee3d6579265004dff4c070bf4844431474.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/b709f18fbd968e94f1a81c674a917cee3d6579265004dff4c070bf4844431474.jpg)

![df5b7eb8c4b148e8713f43904d14c72563bc71b9e1890e3d5e4ef3be8bbeca2d.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/df5b7eb8c4b148e8713f43904d14c72563bc71b9e1890e3d5e4ef3be8bbeca2d.jpg)

![e7e8dce6b6bac2720ae1d040facf1b8585f2ad7b36aa5d7dac37bd3dfafb003a.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/e7e8dce6b6bac2720ae1d040facf1b8585f2ad7b36aa5d7dac37bd3dfafb003a.jpg)

![ef9f6f6dcd949c3cb9d6629395de8a1b6adaa7b53570645be845bbd057dd224d.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/ef9f6f6dcd949c3cb9d6629395de8a1b6adaa7b53570645be845bbd057dd224d.jpg)

![ff0e9f54f9242f9c8331fd659344da1eeee6e99e684bda86fbf7532b720f1e82.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/images/ff0e9f54f9242f9c8331fd659344da1eeee6e99e684bda86fbf7532b720f1e82.jpg)

### Tables

![0592f2152d3daff7306e733d44e9f2ae84cc2d48a592676bed0ea8d92ad066e1.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/0592f2152d3daff7306e733d44e9f2ae84cc2d48a592676bed0ea8d92ad066e1.jpg)

![4db6ca6ccd27ddd0886de15764ed6079644bafdfa31aec259c33bd0015b4ac0a.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/4db6ca6ccd27ddd0886de15764ed6079644bafdfa31aec259c33bd0015b4ac0a.jpg)

![589b54df569a3cbb941e46ab5972db8f3deb08784063566679f423d0925e8516.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/589b54df569a3cbb941e46ab5972db8f3deb08784063566679f423d0925e8516.jpg)

![6e2be13dbe5fa01df435416c6d149dcdb26faba690aa661522f790fa541b0274.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/6e2be13dbe5fa01df435416c6d149dcdb26faba690aa661522f790fa541b0274.jpg)

![b05c5d77308728bf8a5d6b79a7464520179d1660dc6bc798c67ff24e9cf14755.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/b05c5d77308728bf8a5d6b79a7464520179d1660dc6bc798c67ff24e9cf14755.jpg)

![b26f1dfb78f5b4bb4d81db9cb4dadd22dbbc578fdbed3ca50646a0b767ac9547.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/b26f1dfb78f5b4bb4d81db9cb4dadd22dbbc578fdbed3ca50646a0b767ac9547.jpg)

![e4b7bbcfcf617b26a5b5483b04834b8442d1de929ec98dee75564c64c113aba1.jpg](../icml_results/323_Loss-Aware%20Distributionally%20Robust%20Optimization%20via%20Trainable%20Optimal%20Transport%20Ambiguity%20Sets/tables/e4b7bbcfcf617b26a5b5483b04834b8442d1de929ec98dee75564c64c113aba1.jpg)

<a id="paper-0319"></a>

## Linguistic Properties and Model Scale in Brain Encoding: From Small to Compressed Language Models


### Images

![1651bbe6027097014fa8e1fc8695614d171766e113af760effb8e8e103b91c03.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/1651bbe6027097014fa8e1fc8695614d171766e113af760effb8e8e103b91c03.jpg)

![1a1489f0e390dea2778e29e23ac201da36a288939c3670b39dd309b4fb12b65a.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/1a1489f0e390dea2778e29e23ac201da36a288939c3670b39dd309b4fb12b65a.jpg)

![1d6a9e2baa0a2824cbde2494a7ba176b053a201666d3b28ba68c5d25b7c72b44.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/1d6a9e2baa0a2824cbde2494a7ba176b053a201666d3b28ba68c5d25b7c72b44.jpg)

![2872f5854a6164d42caba7774ef96ce86bd151682695abd2681b8073ec701f08.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/2872f5854a6164d42caba7774ef96ce86bd151682695abd2681b8073ec701f08.jpg)

![2b6f74386296d2ffc00fa6a0749bf3f0d9d61b5152a1dc15a9ce4342139f74b0.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/2b6f74386296d2ffc00fa6a0749bf3f0d9d61b5152a1dc15a9ce4342139f74b0.jpg)

![2e22a30fc1c305a13e17f04a932ca28449ba0b9386fcc41d52a706051079c986.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/2e22a30fc1c305a13e17f04a932ca28449ba0b9386fcc41d52a706051079c986.jpg)

![2f168305b404d019924852e5e6a0bc5bb26ea958c1dfc6c1aef2109c4d93c4af.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/2f168305b404d019924852e5e6a0bc5bb26ea958c1dfc6c1aef2109c4d93c4af.jpg)

![31eac7cc89b8e35a6a5a835ed09f42b9e333aab53cbdffe57877a20c8f1aa93f.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/31eac7cc89b8e35a6a5a835ed09f42b9e333aab53cbdffe57877a20c8f1aa93f.jpg)

![360015459e1c30f4c01caa786564ebb074a2f365762931f4da5f5d19735cc167.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/360015459e1c30f4c01caa786564ebb074a2f365762931f4da5f5d19735cc167.jpg)

![3f40e5b57ab6a9b68f1a01d7d32f050010e2580cd8c928ba6afa0fa04e06e5dd.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/3f40e5b57ab6a9b68f1a01d7d32f050010e2580cd8c928ba6afa0fa04e06e5dd.jpg)

![50923d70cdc327e2f5eb7c6fcf58527090d0bdb3cbf21c3d03a093a601e0eb86.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/50923d70cdc327e2f5eb7c6fcf58527090d0bdb3cbf21c3d03a093a601e0eb86.jpg)

![6dbb76a3f5f2f6229670773dd5c227494f0c96fc5d6a97761270810f694c3ac1.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/6dbb76a3f5f2f6229670773dd5c227494f0c96fc5d6a97761270810f694c3ac1.jpg)

![702ea2270e04cafce56b93f37c094bba416aad623ef8dbe440943978f289ba90.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/702ea2270e04cafce56b93f37c094bba416aad623ef8dbe440943978f289ba90.jpg)

![7af1b0b26730d483d2b62e0b572e1b0ea0cf42f0b1d1df13def38131da2613b8.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/7af1b0b26730d483d2b62e0b572e1b0ea0cf42f0b1d1df13def38131da2613b8.jpg)

![80c5e69e0b157062b7d8bf76d0b80fc8838fa6bbd0e532c4c1ee13bd27f25699.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/80c5e69e0b157062b7d8bf76d0b80fc8838fa6bbd0e532c4c1ee13bd27f25699.jpg)

![89b2d6e830e2a42b05cf72077a2ae08e5784adb97f3308567a58c21394532849.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/89b2d6e830e2a42b05cf72077a2ae08e5784adb97f3308567a58c21394532849.jpg)

![8b614aac8bcdaa5b3099c47ef6db1cfed2c0a3a95f5217197d40a0d2e8c542f6.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/8b614aac8bcdaa5b3099c47ef6db1cfed2c0a3a95f5217197d40a0d2e8c542f6.jpg)

![8dee8e19553ffc7e9b6a4585dc201a16bced98ee8ddb12e02c72391e2e4a3d89.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/8dee8e19553ffc7e9b6a4585dc201a16bced98ee8ddb12e02c72391e2e4a3d89.jpg)

![a0d3eeb4018f1487f2d661ed7bb1c7dc38f6a1c782c2247c0e88de7cc2448429.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/a0d3eeb4018f1487f2d661ed7bb1c7dc38f6a1c782c2247c0e88de7cc2448429.jpg)

![ad208b27955b08b8182a2a731b767be0ee1dcdaec8a684bbbf7706c1cc5cc6aa.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/ad208b27955b08b8182a2a731b767be0ee1dcdaec8a684bbbf7706c1cc5cc6aa.jpg)

![ade1bef24890982dd58248fe9b4020834fde12a5103c752eb2b26af889e3f117.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/ade1bef24890982dd58248fe9b4020834fde12a5103c752eb2b26af889e3f117.jpg)

![b31929c732e8ba89e453fae4527591f2e84cd4b97ddc8d887212eb1680240951.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/b31929c732e8ba89e453fae4527591f2e84cd4b97ddc8d887212eb1680240951.jpg)

![bcd4126cddb069156d9d9fcc4ec080ab9b4f2005d79ea59bd49e0319cb98f02d.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/bcd4126cddb069156d9d9fcc4ec080ab9b4f2005d79ea59bd49e0319cb98f02d.jpg)

![bee8abc16340795d270c995e23b4b313403096516a1c11d643084bc8bae008db.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/bee8abc16340795d270c995e23b4b313403096516a1c11d643084bc8bae008db.jpg)

![c8f1cd6d311f1e11cd931360feb86aea9190fb13c1c4d8dcdc5d596a309804ef.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/c8f1cd6d311f1e11cd931360feb86aea9190fb13c1c4d8dcdc5d596a309804ef.jpg)

![d283c77b995e94de6ce1617a206eb10fdde50a5c8856789490404cdf93cc1a69.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/d283c77b995e94de6ce1617a206eb10fdde50a5c8856789490404cdf93cc1a69.jpg)

![dac59e0ff6382ac74676698188df626e90fdb81e8c25a99f227e35273cadec56.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/dac59e0ff6382ac74676698188df626e90fdb81e8c25a99f227e35273cadec56.jpg)

![e1e8942019c4cbe7b1ae3aaceef7c3db9dac45a36c80dc877fa44328d088c255.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/e1e8942019c4cbe7b1ae3aaceef7c3db9dac45a36c80dc877fa44328d088c255.jpg)

![e38b8cad992315b324b86612c50ac5e35b6106890043b8422d5e8e85f1a01901.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/e38b8cad992315b324b86612c50ac5e35b6106890043b8422d5e8e85f1a01901.jpg)

![e72b3a69861e7caac225bee18276d6037f735289b94d3937fa80f5113096952a.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/e72b3a69861e7caac225bee18276d6037f735289b94d3937fa80f5113096952a.jpg)

![e9a2a11f8a0b4cb8c14a79f522f7a5f353c7c1af271ef03ae431241022ca6124.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/e9a2a11f8a0b4cb8c14a79f522f7a5f353c7c1af271ef03ae431241022ca6124.jpg)

![f582f17aead12c5e5786b1df5eec486e67518b74e96352d73ee1f932b037ac25.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/f582f17aead12c5e5786b1df5eec486e67518b74e96352d73ee1f932b037ac25.jpg)

![fa0aac87898ec4831b89d3ccc7b4bcfda46cb19c72aa698dca3757a4e362695c.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/fa0aac87898ec4831b89d3ccc7b4bcfda46cb19c72aa698dca3757a4e362695c.jpg)

![fcdfa3926d09d2e2d48920fcad5db4330c37525d6cc1e7e3b51a456f588bb882.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/images/fcdfa3926d09d2e2d48920fcad5db4330c37525d6cc1e7e3b51a456f588bb882.jpg)

### Tables

![00f2fa5c151a89c3afbd3e40171ac61b1e1996b94119bd63aeedbd459eb5f842.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/00f2fa5c151a89c3afbd3e40171ac61b1e1996b94119bd63aeedbd459eb5f842.jpg)

![06ee7d3d9dee0c7b79de1254bce2be5599d0963b2b6d9f21690ad409295f7ff9.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/06ee7d3d9dee0c7b79de1254bce2be5599d0963b2b6d9f21690ad409295f7ff9.jpg)

![0fd7c25e930f83fa21af27f3dca4feadb16ec41cc08c1f841d86ef8dfdabb94c.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/0fd7c25e930f83fa21af27f3dca4feadb16ec41cc08c1f841d86ef8dfdabb94c.jpg)

![1ea01071398e52cd175a8b6a1f637662ae162b34b97e725f4d9bbaf008e71d49.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/1ea01071398e52cd175a8b6a1f637662ae162b34b97e725f4d9bbaf008e71d49.jpg)

![2930aa9ba34b3641719e2f5a3beccb6107848f1fbc4b1103bcd1edc56ed1d7ed.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/2930aa9ba34b3641719e2f5a3beccb6107848f1fbc4b1103bcd1edc56ed1d7ed.jpg)

![29b5394cb8ac9ad21b1d8710f4f1e1b83b35ebb3064c78c016ccd6183e0ed60e.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/29b5394cb8ac9ad21b1d8710f4f1e1b83b35ebb3064c78c016ccd6183e0ed60e.jpg)

![39de3854bbae464b0b42812c91302d27776826dc4e4667fbd01762d243231cd8.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/39de3854bbae464b0b42812c91302d27776826dc4e4667fbd01762d243231cd8.jpg)

![42a872b44a4310aea665ab1cdfa9e3e1eb879cb70c8b82799e66c5503978abff.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/42a872b44a4310aea665ab1cdfa9e3e1eb879cb70c8b82799e66c5503978abff.jpg)

![4d7af74ec5c372ad0e6fc808711c609c9eec10f0fc46a77339de13aa227e3e1e.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/4d7af74ec5c372ad0e6fc808711c609c9eec10f0fc46a77339de13aa227e3e1e.jpg)

![64014ac269003b8ae8bd69d3cb9c48dbda80df0fe34f90e2f7967e82f4595446.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/64014ac269003b8ae8bd69d3cb9c48dbda80df0fe34f90e2f7967e82f4595446.jpg)

![648d1b19187bbbfdf636bb3129afcdedc2493ac16a1fc8bfe63307c63f496465.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/648d1b19187bbbfdf636bb3129afcdedc2493ac16a1fc8bfe63307c63f496465.jpg)

![65d5ca14bcecd25fc09e376669ef4f5bd47471d586be1c5ba37d294b3964a984.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/65d5ca14bcecd25fc09e376669ef4f5bd47471d586be1c5ba37d294b3964a984.jpg)

![6760087799f2686490799b9fc07eb2459d0fb8ee3d7b521937056cc68b8c1fa1.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/6760087799f2686490799b9fc07eb2459d0fb8ee3d7b521937056cc68b8c1fa1.jpg)

![74342451ddcf21ee54fb62f8dd0d74051840e7ccc97e3925014ab815ce7216a5.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/74342451ddcf21ee54fb62f8dd0d74051840e7ccc97e3925014ab815ce7216a5.jpg)

![7697219dd5c836ec1d90efef922b98dcd464099c5349ca25bc0b5ac488ae05c1.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/7697219dd5c836ec1d90efef922b98dcd464099c5349ca25bc0b5ac488ae05c1.jpg)

![802ad6995db0d4d1415ebc36e75049e7a0bf84bd1f263f016e6454b4d43e9f99.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/802ad6995db0d4d1415ebc36e75049e7a0bf84bd1f263f016e6454b4d43e9f99.jpg)

![90f158638373203b7e3c60b9823228fada49ac7cedd127cde3a335ad267fab6c.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/90f158638373203b7e3c60b9823228fada49ac7cedd127cde3a335ad267fab6c.jpg)

![90ff5f510fc4f3fda89b728ee795cad08ec4344afb8e26b09fc78240dfd7161f.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/90ff5f510fc4f3fda89b728ee795cad08ec4344afb8e26b09fc78240dfd7161f.jpg)

![9ae04d364490d4900bce3b64700e3ed94a86e8e840ca4b73143b0917a3cfc9d1.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/9ae04d364490d4900bce3b64700e3ed94a86e8e840ca4b73143b0917a3cfc9d1.jpg)

![a0bbb8d79fda062ddc876a78a70eb905eaf836dcee3e9393e962023952d6e405.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/a0bbb8d79fda062ddc876a78a70eb905eaf836dcee3e9393e962023952d6e405.jpg)

![ab6b3cef04b40d6996b620d83b26637042ac0eab89257318128b722bc8204711.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/ab6b3cef04b40d6996b620d83b26637042ac0eab89257318128b722bc8204711.jpg)

![adbae60a4c284110b8996d7353f329e2607c3381482f52662899f67edefdd3d2.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/adbae60a4c284110b8996d7353f329e2607c3381482f52662899f67edefdd3d2.jpg)

![c1ee07c1325a24f9b016f490c4d397cbcdf4f16b20b5d9a0832c6f1c88179984.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/c1ee07c1325a24f9b016f490c4d397cbcdf4f16b20b5d9a0832c6f1c88179984.jpg)

![ded41a017cb23da1c9760f40c117017aba0a9189b370191371cece97316137ad.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/ded41a017cb23da1c9760f40c117017aba0a9189b370191371cece97316137ad.jpg)

![ee999a0a374d1f288e5692d669105159d0c9f613d74d9cc4a4fc9d91b2d883ff.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/ee999a0a374d1f288e5692d669105159d0c9f613d74d9cc4a4fc9d91b2d883ff.jpg)

![fac7a5c1e76812aa7f4437b40678f40e29fdf4da6923f4efe31c2b2621df34d6.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/fac7a5c1e76812aa7f4437b40678f40e29fdf4da6923f4efe31c2b2621df34d6.jpg)

![fcf6c92dd77a3d4f435cb878b37f5a397ae54cb8e925db26dcb68a6179a32122.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/fcf6c92dd77a3d4f435cb878b37f5a397ae54cb8e925db26dcb68a6179a32122.jpg)

![ff72b82b5e5a84e7a9d5faae3de8e00110f75b6f047b92688c8efcb6d2f59991.jpg](../icml_results/324_Linguistic%20Properties%20and%20Model%20Scale%20in%20Brain%20Encoding_%20From%20Small%20to%20Compressed%20Language%20Models/tables/ff72b82b5e5a84e7a9d5faae3de8e00110f75b6f047b92688c8efcb6d2f59991.jpg)

<a id="paper-0320"></a>

## Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities


### Images

![85f66698d0c487a847253511175ba101b6246e3b9b7f64d38b3b480b3e6c9801.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/images/85f66698d0c487a847253511175ba101b6246e3b9b7f64d38b3b480b3e6c9801.jpg)

![f36da8f2d9a6a4620fff6d4bda033a9397fd470d5dcf6dbc66470f3d449346f6.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/images/f36da8f2d9a6a4620fff6d4bda033a9397fd470d5dcf6dbc66470f3d449346f6.jpg)

### Tables

![1d84620ce3288d4fa8b178740b1f74b0e7de5d89344749a45ec458b6cfcb0b88.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/1d84620ce3288d4fa8b178740b1f74b0e7de5d89344749a45ec458b6cfcb0b88.jpg)

![25803726d325108107a5f0da3aa57bf42bee1bed499746ef40e5711ed909016d.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/25803726d325108107a5f0da3aa57bf42bee1bed499746ef40e5711ed909016d.jpg)

![38d8c748e91a4ebdfe595bd601cc14e9cf66c6673a80c812fc49c32027238a95.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/38d8c748e91a4ebdfe595bd601cc14e9cf66c6673a80c812fc49c32027238a95.jpg)

![587e2ac2cb2e5264d85ae5fe7c3efa69412ed0b80dbd69a5849d16751f44d010.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/587e2ac2cb2e5264d85ae5fe7c3efa69412ed0b80dbd69a5849d16751f44d010.jpg)

![62cd066c5fc7ba334db877155490164e5946ab61369929a991faaa0f1e25a5b8.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/62cd066c5fc7ba334db877155490164e5946ab61369929a991faaa0f1e25a5b8.jpg)

![7a43bb7f9a1559d2a3bf074f46258f3ac21652f5af2d8722df9f1525397dc912.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/7a43bb7f9a1559d2a3bf074f46258f3ac21652f5af2d8722df9f1525397dc912.jpg)

![82bddd1c820bf2428f486817788bfd6325895f508ea404943cf7de16b784fabf.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/82bddd1c820bf2428f486817788bfd6325895f508ea404943cf7de16b784fabf.jpg)

![c6249633b30a3c81c5437959e1995f67b950f9b572c5ddb391f4abf3eaeeee82.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/c6249633b30a3c81c5437959e1995f67b950f9b572c5ddb391f4abf3eaeeee82.jpg)

![f4dbb2fab8c014f6d05364fd88900a8e60a943e8ee626d0d8a5c2c3905a13cfe.jpg](../icml_results/325_Updating%20Parametric%20Knowledge%20with%20Context%20Distillation%20Retains%20Post-Training%20Capabilities/tables/f4dbb2fab8c014f6d05364fd88900a8e60a943e8ee626d0d8a5c2c3905a13cfe.jpg)

<a id="paper-0321"></a>

## Measuring Agents in Production


### Images

![081d9bfe57f32230cc9e0a8aebdb5684e194a9b5b1db7cf1a0a17bfcef28ac0e.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/081d9bfe57f32230cc9e0a8aebdb5684e194a9b5b1db7cf1a0a17bfcef28ac0e.jpg)

![155493f7034818cddc57cbaa948549f4360686edbcffa46f4b6546db53344460.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/155493f7034818cddc57cbaa948549f4360686edbcffa46f4b6546db53344460.jpg)

![190b3ac01dba31096c11589bbc0d96328d246d624d4900d8f5b9de44a141e865.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/190b3ac01dba31096c11589bbc0d96328d246d624d4900d8f5b9de44a141e865.jpg)

![22a9ee02571df16e3eb028eda2df7a5e3192053ac742fd44302afdd9a4639cad.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/22a9ee02571df16e3eb028eda2df7a5e3192053ac742fd44302afdd9a4639cad.jpg)

![37455d72a6753193b8b8a45fdf678680bf8cd72a7df22abf416a309e37b1a090.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/37455d72a6753193b8b8a45fdf678680bf8cd72a7df22abf416a309e37b1a090.jpg)

![4144554ced88263b0021515872559980eecfbf3a1104f35b73e03a10da713ace.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/4144554ced88263b0021515872559980eecfbf3a1104f35b73e03a10da713ace.jpg)

![440272d90869f6e4395b7cb1ff057a10dcf28fdb990b020af726af4125f57d5f.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/440272d90869f6e4395b7cb1ff057a10dcf28fdb990b020af726af4125f57d5f.jpg)

![4f130237d3513638775c4d2a69c6392610e9e534479bbbd08d7950278ce94ff7.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/4f130237d3513638775c4d2a69c6392610e9e534479bbbd08d7950278ce94ff7.jpg)

![53472783d22a67be095736c6984d7691939ba482f2069b80f7768e5b1d3cc487.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/53472783d22a67be095736c6984d7691939ba482f2069b80f7768e5b1d3cc487.jpg)

![62ed79c4ef991d5909ada2cd677a1365eb9dd28e12ab96dbd5130cc0c898a6c6.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/62ed79c4ef991d5909ada2cd677a1365eb9dd28e12ab96dbd5130cc0c898a6c6.jpg)

![63f2333e9c761ed0125a5adb1ae8ec5fe3a0390ffe20ef7d3347b115e1b433db.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/63f2333e9c761ed0125a5adb1ae8ec5fe3a0390ffe20ef7d3347b115e1b433db.jpg)

![6bde062de8ff05d3890bc8d9e2ea1421b0ededffefa74707f338ff60535ac65a.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/6bde062de8ff05d3890bc8d9e2ea1421b0ededffefa74707f338ff60535ac65a.jpg)

![757e734baa5699085879bcf8157b858064dc4f68744b35e9b13e82f21f8ac043.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/757e734baa5699085879bcf8157b858064dc4f68744b35e9b13e82f21f8ac043.jpg)

![766d9db6c27bb7b18ac822f61661eceda9fcbf8fb4537ae3e5926e04ddd123b3.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/766d9db6c27bb7b18ac822f61661eceda9fcbf8fb4537ae3e5926e04ddd123b3.jpg)

![7c450d531f46b4538d31f02a32aaec893f366d39edea378ac1315e6c12a478b6.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/7c450d531f46b4538d31f02a32aaec893f366d39edea378ac1315e6c12a478b6.jpg)

![845b0b75d10c9cd53a8090bcfa8c87f31adbeec40dc95213b2efe28958e3870f.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/845b0b75d10c9cd53a8090bcfa8c87f31adbeec40dc95213b2efe28958e3870f.jpg)

![86c91bd94f1f2c24339c88b9421c1a0fe722a52a15dc7d365efbe0023b246d8d.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/86c91bd94f1f2c24339c88b9421c1a0fe722a52a15dc7d365efbe0023b246d8d.jpg)

![86f3d5bdcff92f376410fcb528e1825678cf5fa3fd86f41a701659a8d5d9b411.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/86f3d5bdcff92f376410fcb528e1825678cf5fa3fd86f41a701659a8d5d9b411.jpg)

![a71185630d0337ab7660363688d50e6a507a53e20393c75c5195f2e1a30870d3.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/a71185630d0337ab7660363688d50e6a507a53e20393c75c5195f2e1a30870d3.jpg)

![a8fb4d9e6457ee9e395896da2cf18467f85b655e8e6da0069ab336963cd9bcec.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/a8fb4d9e6457ee9e395896da2cf18467f85b655e8e6da0069ab336963cd9bcec.jpg)

![aa7f1ec53a3c1c6d777f02117b0d7d7c2a8e38e7a2cadd123b2f2951bb5230fa.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/aa7f1ec53a3c1c6d777f02117b0d7d7c2a8e38e7a2cadd123b2f2951bb5230fa.jpg)

![ac2f07aed02ce76243176b3eda4809001f50cbf2b9ee7953ff86fd76311b7190.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/ac2f07aed02ce76243176b3eda4809001f50cbf2b9ee7953ff86fd76311b7190.jpg)

![b05e32729265c8e92c36074bbd498ec3878c1c65aaa493ffb438024140bf7876.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/b05e32729265c8e92c36074bbd498ec3878c1c65aaa493ffb438024140bf7876.jpg)

![b4d8dc2fe73f8b6c33e8bc7756924a08215744fe0d5934f34e65e519d83e2bd5.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/b4d8dc2fe73f8b6c33e8bc7756924a08215744fe0d5934f34e65e519d83e2bd5.jpg)

![b68acde137bff579e55b0ae588333d09b36cc74732cdbd5f7b38c0442f6263b1.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/b68acde137bff579e55b0ae588333d09b36cc74732cdbd5f7b38c0442f6263b1.jpg)

![bc1ca25bf2c9a46ea8c14cdd2b03733dbc154f413026a96151f24e48b6295c2e.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/bc1ca25bf2c9a46ea8c14cdd2b03733dbc154f413026a96151f24e48b6295c2e.jpg)

![ca0a93caa29a37aa3fd5986b50d8cb2192405fbba2c475cf68d9687a71a0b3d7.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/ca0a93caa29a37aa3fd5986b50d8cb2192405fbba2c475cf68d9687a71a0b3d7.jpg)

![ec823e890bc88ba26f636be0a7fee8ac1320f93a1feae5d2c3c90706f3e471b6.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/ec823e890bc88ba26f636be0a7fee8ac1320f93a1feae5d2c3c90706f3e471b6.jpg)

![f5c6b930c73aed876fe7c39791087560a3edd7a12c46ef0a2af1bc37b045d7d7.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/f5c6b930c73aed876fe7c39791087560a3edd7a12c46ef0a2af1bc37b045d7d7.jpg)

![fff2100cf1cebdfe4a2e6899978f82151606aa15c0e4452963901bb44f05dac4.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/images/fff2100cf1cebdfe4a2e6899978f82151606aa15c0e4452963901bb44f05dac4.jpg)

### Tables

![036f484533bed79a28841da046f0d43a56c283fa6a27dd47c547047d65bc87c3.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/036f484533bed79a28841da046f0d43a56c283fa6a27dd47c547047d65bc87c3.jpg)

![051ac21ab2cdd13922ce8c38f4c19d75b6a7ba6780b1c217bd4e28e0bbf3f502.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/051ac21ab2cdd13922ce8c38f4c19d75b6a7ba6780b1c217bd4e28e0bbf3f502.jpg)

![0c67a3458befbe8823ab6b46536e9ba3fb041114485ee0c1402ebcb1a628ad0a.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/0c67a3458befbe8823ab6b46536e9ba3fb041114485ee0c1402ebcb1a628ad0a.jpg)

![0ded69a81e050d45e6c8d8ebdff2cd03bdf6846e1184e19761ff74b4c518925d.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/0ded69a81e050d45e6c8d8ebdff2cd03bdf6846e1184e19761ff74b4c518925d.jpg)

![3112d3ac2e65975d5516bbecff94d22aace8ce4b20ba7ca372fe30e82b2aebb9.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/3112d3ac2e65975d5516bbecff94d22aace8ce4b20ba7ca372fe30e82b2aebb9.jpg)

![3b211cfd283a777ae0fd141a4074609e73e7bf3ad027b8c36f6a0dfc94b4c055.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/3b211cfd283a777ae0fd141a4074609e73e7bf3ad027b8c36f6a0dfc94b4c055.jpg)

![3ca4cf3692575d30447b3d6d84285b638cc8584715510d00c722162e97e90e1e.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/3ca4cf3692575d30447b3d6d84285b638cc8584715510d00c722162e97e90e1e.jpg)

![3f624f73f0301c42c45b03f23f50f36f386563956a93276ecfb8b2ddeeb4306e.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/3f624f73f0301c42c45b03f23f50f36f386563956a93276ecfb8b2ddeeb4306e.jpg)

![67feb2fa2f074734c8edc75e39938e85cf6b245e7025aa1888563a58bf019c16.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/67feb2fa2f074734c8edc75e39938e85cf6b245e7025aa1888563a58bf019c16.jpg)

![9736f6fc3924f3ae438c53754d8105bb1560248d6106faffde8902bdaee0c77b.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/9736f6fc3924f3ae438c53754d8105bb1560248d6106faffde8902bdaee0c77b.jpg)

![97c9f7e1680d869b926453beb82b5a769cf1a520e5a9b76ef8aa56769c7d77ed.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/97c9f7e1680d869b926453beb82b5a769cf1a520e5a9b76ef8aa56769c7d77ed.jpg)

![a46a3f10fe9f5a1e3797fa1ab8d002ab7c38f46897f457ee3690522f1345b69b.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/a46a3f10fe9f5a1e3797fa1ab8d002ab7c38f46897f457ee3690522f1345b69b.jpg)

![b7978a71d1ce278c5749de496b1bdb63584d30a646816f36bdd8917304e9d876.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/b7978a71d1ce278c5749de496b1bdb63584d30a646816f36bdd8917304e9d876.jpg)

![df4ad1f5a299460b74c8f0ad0fdb57f975ddb4a63412d4b1624d5d5170f2b806.jpg](../icml_results/326_Measuring%20Agents%20in%20Production/tables/df4ad1f5a299460b74c8f0ad0fdb57f975ddb4a63412d4b1624d5d5170f2b806.jpg)

<a id="paper-0322"></a>

## Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs


### Images

![09816138e637c6111d6ec2d518f37c64d5fc169aa55c43600ec06b1c648ab604.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/09816138e637c6111d6ec2d518f37c64d5fc169aa55c43600ec06b1c648ab604.jpg)

![46222e49a0ca00e39889d57feb724c2d47105e2fa744b81b9c590f0b50dee7db.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/46222e49a0ca00e39889d57feb724c2d47105e2fa744b81b9c590f0b50dee7db.jpg)

![4c11f2de9d50058e393b56752669b570b67022a2e53e1fbaad043e4f512fad56.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/4c11f2de9d50058e393b56752669b570b67022a2e53e1fbaad043e4f512fad56.jpg)

![7668fa975fe88411e680f6209cace32dc872699e7c76d17b0497c2991e4be6f8.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/7668fa975fe88411e680f6209cace32dc872699e7c76d17b0497c2991e4be6f8.jpg)

![93ae8363045d0c3ae0029c56f0b9676ea1cdb8751a0bc1c14a9e47222dc96063.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/93ae8363045d0c3ae0029c56f0b9676ea1cdb8751a0bc1c14a9e47222dc96063.jpg)

![c7c800ab8b470c3045db8c55ceed0e1b1eb411eb4a852724e7110f124d5f8198.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/c7c800ab8b470c3045db8c55ceed0e1b1eb411eb4a852724e7110f124d5f8198.jpg)

![fdd8262b85b3c4923f503be2ce43fcfff0942cd49e25aeda2c542e9953150dab.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/images/fdd8262b85b3c4923f503be2ce43fcfff0942cd49e25aeda2c542e9953150dab.jpg)

### Tables

![27c9d14abfccfe93995fae593e834d74415ba117824e2df837d3744991ebfb44.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/27c9d14abfccfe93995fae593e834d74415ba117824e2df837d3744991ebfb44.jpg)

![399ecf237d6e9e83194b41eea79c6a9f86f75abb98d5e290b8719fc2a5453fae.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/399ecf237d6e9e83194b41eea79c6a9f86f75abb98d5e290b8719fc2a5453fae.jpg)

![4b14d2217d0a0c7e68203ff67bf4dbb7465ce49394fa84c570c29230217e3bfe.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/4b14d2217d0a0c7e68203ff67bf4dbb7465ce49394fa84c570c29230217e3bfe.jpg)

![58c293fdfea524c41a1d20285b3cb1f1ee8f628715372af1e8818e46e6f944bb.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/58c293fdfea524c41a1d20285b3cb1f1ee8f628715372af1e8818e46e6f944bb.jpg)

![5ac1abafd3b5d5972b7a2b1dee7da9000743b6e946c45761ea47a428dc3c49ba.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/5ac1abafd3b5d5972b7a2b1dee7da9000743b6e946c45761ea47a428dc3c49ba.jpg)

![72fbf70ef9b607f5c0dc4ede04aabbadb2ed7a0a03eaff94da4a33050fb5a83b.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/72fbf70ef9b607f5c0dc4ede04aabbadb2ed7a0a03eaff94da4a33050fb5a83b.jpg)

![7e923c07510c242cb284f38d7830909a1f777494f4bab1f6ac65fb9e65fbcc70.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/7e923c07510c242cb284f38d7830909a1f777494f4bab1f6ac65fb9e65fbcc70.jpg)

![82371eb359e391030d546d44f771a165592366b8f380cb060c8aa801fef4836a.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/82371eb359e391030d546d44f771a165592366b8f380cb060c8aa801fef4836a.jpg)

![8ef9610fec2c9aaff32fcd6d09ba599608d30920caf4938dbc270f74f3fbb7a1.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/8ef9610fec2c9aaff32fcd6d09ba599608d30920caf4938dbc270f74f3fbb7a1.jpg)

![d5e4e804285c86c9d32c07d866c75105d4cf8e1a0c5138c5451fef23510f83e8.jpg](../icml_results/327_Towards%20Long-Horizon%20Interpretability_%20Efficient%20and%20Faithful%20Multi-Token%20Attribution%20for%20Reasoning%20/tables/d5e4e804285c86c9d32c07d866c75105d4cf8e1a0c5138c5451fef23510f83e8.jpg)

<a id="paper-0323"></a>

## Adaptive Policy Backbone via Shared Network


### Images

![0222de1c68c3f15e39ba6cf6d5d6352db13fef2156f0cccd60df42ae8a764c46.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/0222de1c68c3f15e39ba6cf6d5d6352db13fef2156f0cccd60df42ae8a764c46.jpg)

![2eed432d413890cb625b15d70905adf754f829b35aa8654725547ae741896894.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/2eed432d413890cb625b15d70905adf754f829b35aa8654725547ae741896894.jpg)

![317cf20ea6321bdbaf09b2e14203f03872cfcd3fdd03d9885f715e6ed76dd60c.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/317cf20ea6321bdbaf09b2e14203f03872cfcd3fdd03d9885f715e6ed76dd60c.jpg)

![3b1eaaf58f7b18ee3466dbbccae9ff3d678975d0f1fd5858b992c9fd753d4a80.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/3b1eaaf58f7b18ee3466dbbccae9ff3d678975d0f1fd5858b992c9fd753d4a80.jpg)

![3b72b7d7bcca49f291e7dcd5f79ae076258475a6f07f699c17d1b43d918bbc50.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/3b72b7d7bcca49f291e7dcd5f79ae076258475a6f07f699c17d1b43d918bbc50.jpg)

![57dd3380496164189e45b558a9229b1caf956eca537563f2af6a2c7297696d6c.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/57dd3380496164189e45b558a9229b1caf956eca537563f2af6a2c7297696d6c.jpg)

![874847ccb4a7315b75da9b7cbab176c03ac4d3ea41379d3f6606d7fde5231270.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/874847ccb4a7315b75da9b7cbab176c03ac4d3ea41379d3f6606d7fde5231270.jpg)

![da60b54c87f1cb3a9d5dc37ffffbb4c4b3bb1a366445df2941746a4b1d6037fd.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/images/da60b54c87f1cb3a9d5dc37ffffbb4c4b3bb1a366445df2941746a4b1d6037fd.jpg)

### Tables

![0886ca737cb06d964e57ad3e9c5efa3e9711e4308015ab32e7d88eb94ddcf93d.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/tables/0886ca737cb06d964e57ad3e9c5efa3e9711e4308015ab32e7d88eb94ddcf93d.jpg)

![52fafade954fdb009c2793ccd1f4babe8b52c5e48143c6e6edc196aca4cfee10.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/tables/52fafade954fdb009c2793ccd1f4babe8b52c5e48143c6e6edc196aca4cfee10.jpg)

![fa96ec36b849c08f966ac2db7f8b1e57627f11a79e1009d86ae4d62567117ee1.jpg](../icml_results/328_Adaptive%20Policy%20Backbone%20via%20Shared%20Network/tables/fa96ec36b849c08f966ac2db7f8b1e57627f11a79e1009d86ae4d62567117ee1.jpg)

<a id="paper-0324"></a>

## On Minimum Depth and Width of Floating-Point Neural Networks for Representing Floating-Point Functions


### Tables

![2fd9d1262261bde680035b74d84c77e76f7e17097d2c15ce726a0cb3a0631312.jpg](../icml_results/329_On%20Minimum%20Depth%20and%20Width%20of%20Floating-Point%20Neural%20Networks%20for%20Representing%20Floating-Point%20Functio/tables/2fd9d1262261bde680035b74d84c77e76f7e17097d2c15ce726a0cb3a0631312.jpg)

![98f6b784d399b4920ef0133d4655fc4287c842552e4ca0324b73793a00bf8be5.jpg](../icml_results/329_On%20Minimum%20Depth%20and%20Width%20of%20Floating-Point%20Neural%20Networks%20for%20Representing%20Floating-Point%20Functio/tables/98f6b784d399b4920ef0133d4655fc4287c842552e4ca0324b73793a00bf8be5.jpg)

<a id="paper-0325"></a>

## Stable Deep Reinforcement Learning via Isotropic Gaussian Representations


### Images

![0fec91697c15f7ab2a2b51fe2a82104922e35a735f990ed5ffcbf5cae0da7326.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/0fec91697c15f7ab2a2b51fe2a82104922e35a735f990ed5ffcbf5cae0da7326.jpg)

![1118250c1fdc6b70277afd179808e7bd3bfcc889932e500c7847f4385c2e862c.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/1118250c1fdc6b70277afd179808e7bd3bfcc889932e500c7847f4385c2e862c.jpg)

![1a2cff0d6a31a22cbd36f3f4d07b3eeece8594fe6af114d243a91f2deda659be.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/1a2cff0d6a31a22cbd36f3f4d07b3eeece8594fe6af114d243a91f2deda659be.jpg)

![259869ed94ff06bf6fe187ec812cf022fddf32c10a03e3c9c05b4beafb07f414.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/259869ed94ff06bf6fe187ec812cf022fddf32c10a03e3c9c05b4beafb07f414.jpg)

![284134f58a23be6a69582e7b55faad0d1a17110e9f8e0f1c5e40ecbb84cc8046.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/284134f58a23be6a69582e7b55faad0d1a17110e9f8e0f1c5e40ecbb84cc8046.jpg)

![2b276318cc3530a6321aa2097bef8253d7184467ca93b09e827f5c7be9e585d9.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/2b276318cc3530a6321aa2097bef8253d7184467ca93b09e827f5c7be9e585d9.jpg)

![2d1010d030368cfc6f63ab2ae2cff4e8b12e2905e2f62297369e25c7b8fb73bf.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/2d1010d030368cfc6f63ab2ae2cff4e8b12e2905e2f62297369e25c7b8fb73bf.jpg)

![2eda7da44ef85dd2bc5ad9202c0a75a54aee76233cca16e30d1458b469e2a5cf.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/2eda7da44ef85dd2bc5ad9202c0a75a54aee76233cca16e30d1458b469e2a5cf.jpg)

![38117105655003bba4bfd09cfdc68d9f86eeb0ff09eaa090e2f3a3a78b1677b1.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/38117105655003bba4bfd09cfdc68d9f86eeb0ff09eaa090e2f3a3a78b1677b1.jpg)

![3977c034eb7dabd72ebacc7046096a348f0ed29dd044e075dc9a0f24b3535479.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/3977c034eb7dabd72ebacc7046096a348f0ed29dd044e075dc9a0f24b3535479.jpg)

![3f24668c643994081ca60dc8fe9bd60ddd99ae6fba7590d9f825c28853a64dd0.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/3f24668c643994081ca60dc8fe9bd60ddd99ae6fba7590d9f825c28853a64dd0.jpg)

![4028b335ebb01db13323428a657df520a6a23e3383fd4655d1c9fd9c331b6327.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/4028b335ebb01db13323428a657df520a6a23e3383fd4655d1c9fd9c331b6327.jpg)

![417bfae01af8cf7deab9f3694f3ced0d2e5a938544cda5f8b2760d56b76fc7f4.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/417bfae01af8cf7deab9f3694f3ced0d2e5a938544cda5f8b2760d56b76fc7f4.jpg)

![43cc9f386f4ca446433d4474d102aa3150cfbae8611f490c27ce33ddaf14fcb2.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/43cc9f386f4ca446433d4474d102aa3150cfbae8611f490c27ce33ddaf14fcb2.jpg)

![52cb4eb51103704518050011690967e99140ccbd34a0c1683b587118df850d1f.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/52cb4eb51103704518050011690967e99140ccbd34a0c1683b587118df850d1f.jpg)

![572a8dfdbbec6840deb122a757fc34cf108cea7de5b8596a1f43eee167fba628.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/572a8dfdbbec6840deb122a757fc34cf108cea7de5b8596a1f43eee167fba628.jpg)

![5a4b1c4786e64c4fd1b907ab44b8fdef8174aae5229db2f438eb0ca52b650ee2.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/5a4b1c4786e64c4fd1b907ab44b8fdef8174aae5229db2f438eb0ca52b650ee2.jpg)

![6388d26ee927a61a38d24d7e707c21a59cab6bbf837f81bf77212d20ecc9c687.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/6388d26ee927a61a38d24d7e707c21a59cab6bbf837f81bf77212d20ecc9c687.jpg)

![7f85bef6e67a926e67d56ef37372e3068c2ca227312d24c4b7049181267b1558.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/7f85bef6e67a926e67d56ef37372e3068c2ca227312d24c4b7049181267b1558.jpg)

![8703014195d59f2189fe1d82faa90bdc0785cf28f0916a49b95f82a02ad7ddb3.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/8703014195d59f2189fe1d82faa90bdc0785cf28f0916a49b95f82a02ad7ddb3.jpg)

![89d1722416dafa7788440e146f28a6f8ee53851297c6cf66569c0d33aa91d89b.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/89d1722416dafa7788440e146f28a6f8ee53851297c6cf66569c0d33aa91d89b.jpg)

![8a1d02d446fc2d67a3b34f8f78c1eb90f9154d1443c93a9f45883f79ac157844.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/8a1d02d446fc2d67a3b34f8f78c1eb90f9154d1443c93a9f45883f79ac157844.jpg)

![915ba291814b98996e1ef068096e4a704dbf17eb2a86b75262c22149b4564e42.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/915ba291814b98996e1ef068096e4a704dbf17eb2a86b75262c22149b4564e42.jpg)

![9c8e3371850b0f2ac080d6f4b6202d2d19db5a4a31b22c902152c4d772a29a5f.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/9c8e3371850b0f2ac080d6f4b6202d2d19db5a4a31b22c902152c4d772a29a5f.jpg)

![a6e3ad68bdc79c3a0d68a8c371d9268db336be1e622af06aa5ef1fd3cb8a4294.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/a6e3ad68bdc79c3a0d68a8c371d9268db336be1e622af06aa5ef1fd3cb8a4294.jpg)

![acdde7ff2f13221037ea22c255c8669b16b2561f17cda49118169dc4d3551d5c.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/acdde7ff2f13221037ea22c255c8669b16b2561f17cda49118169dc4d3551d5c.jpg)

![c6c2bcd9c826db0cd80e7aaa334acb4affd7566e80bf964fcc20765dbaaed14b.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/c6c2bcd9c826db0cd80e7aaa334acb4affd7566e80bf964fcc20765dbaaed14b.jpg)

![cce6ef7c409c53551946631944509a92e109ec15ee2d7d8b8f82c62976431c6b.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/cce6ef7c409c53551946631944509a92e109ec15ee2d7d8b8f82c62976431c6b.jpg)

![d9be71cdc938b8b4dd7fe447b0626e258721380916fea5cdf64d9f08c3146630.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/d9be71cdc938b8b4dd7fe447b0626e258721380916fea5cdf64d9f08c3146630.jpg)

![dec9a4ac7b73f2108e433d2caf5655b3afe0bc35cb47cecc30a3869fb3452886.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/dec9a4ac7b73f2108e433d2caf5655b3afe0bc35cb47cecc30a3869fb3452886.jpg)

![e3ac7f8d81e0a86b6bb31ec83f8d2f121c24fadfd19d23d9728a31342ee2a8b4.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/e3ac7f8d81e0a86b6bb31ec83f8d2f121c24fadfd19d23d9728a31342ee2a8b4.jpg)

![e534108cf415efeb574a9d471aba9c7cfd42b23622e1258a6a7f39d1064664ff.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/e534108cf415efeb574a9d471aba9c7cfd42b23622e1258a6a7f39d1064664ff.jpg)

![f7ee0c3e7e63f0fb86c90595d20d0539fb56d3cd4b2affb1b3d509fab609089d.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/images/f7ee0c3e7e63f0fb86c90595d20d0539fb56d3cd4b2affb1b3d509fab609089d.jpg)

### Tables

![0862a59138bfb44f225b24fde9bede32953e4971f845bcbb61b158ba0d400d80.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/0862a59138bfb44f225b24fde9bede32953e4971f845bcbb61b158ba0d400d80.jpg)

![1233411a287a3a2b3dc5fed5ec1152dd02785f238b029da3c930b9ba3b2c60e9.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/1233411a287a3a2b3dc5fed5ec1152dd02785f238b029da3c930b9ba3b2c60e9.jpg)

![2cad8f0be20ef04e5b720b10b658b4e656db8fb9c77f5a9ee942ce0f41cb07d6.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/2cad8f0be20ef04e5b720b10b658b4e656db8fb9c77f5a9ee942ce0f41cb07d6.jpg)

![646b767255480a895d993ae533f1ccc5b8091478144b500554b00a8b6f4abd81.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/646b767255480a895d993ae533f1ccc5b8091478144b500554b00a8b6f4abd81.jpg)

![798f404cac9ef1fd3f593d67540075aa6e19a00d97c662e00c73097d43434f45.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/798f404cac9ef1fd3f593d67540075aa6e19a00d97c662e00c73097d43434f45.jpg)

![7c257afd60e987a49abf3c35905724a7376bb2f1f5310bf68f6679a0b8a63102.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/7c257afd60e987a49abf3c35905724a7376bb2f1f5310bf68f6679a0b8a63102.jpg)

![960d3f7fb2c67e4e44d8027c6675370d54f1f485a4ce824ca1f49ee17838ebf7.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/960d3f7fb2c67e4e44d8027c6675370d54f1f485a4ce824ca1f49ee17838ebf7.jpg)

![a048c7acd7a940ac5ec9577c456beb98ab83323aac27def1c9858e5cf3d91355.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/a048c7acd7a940ac5ec9577c456beb98ab83323aac27def1c9858e5cf3d91355.jpg)

![ab3e61739877736aefa4a6f27ae9c06a185380acd17abb269a45bab95657b39e.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/ab3e61739877736aefa4a6f27ae9c06a185380acd17abb269a45bab95657b39e.jpg)

![cd3155a8cbfec071dcb06d33b8fc37ccb8c81ad88b0e0aa5f1b879b6ce6e9f40.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/cd3155a8cbfec071dcb06d33b8fc37ccb8c81ad88b0e0aa5f1b879b6ce6e9f40.jpg)

![ce9159592afebc5551d3ddc67fcc6ff1fbc76ac523bc30ce07329e6c2cd43cc6.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/ce9159592afebc5551d3ddc67fcc6ff1fbc76ac523bc30ce07329e6c2cd43cc6.jpg)

![e6a4fc99943cbf61d0f73ff25f819321b1b1d4929eb8555431d9a621abf8d586.jpg](../icml_results/330_Stable%20Deep%20Reinforcement%20Learning%20via%20Isotropic%20Gaussian%20Representations/tables/e6a4fc99943cbf61d0f73ff25f819321b1b1d4929eb8555431d9a621abf8d586.jpg)

<a id="paper-0326"></a>

## SoftJAX & SoftTorch: Empowering Automatic Differentiation Libraries with Informative Gradients


### Images

![0d8f111420a0e86826bdb9697b5f70b8352845d5356251727f37ef55b52aef95.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/0d8f111420a0e86826bdb9697b5f70b8352845d5356251727f37ef55b52aef95.jpg)

![153bf31d831bba3792dd5185e36a6e54f1ac3f0091c6ed3db83a1c3eb569d4d1.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/153bf31d831bba3792dd5185e36a6e54f1ac3f0091c6ed3db83a1c3eb569d4d1.jpg)

![16c9edc4cb588c9e997a07fe4bc81f6f6bd323801dc898371c2dbcda2082772e.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/16c9edc4cb588c9e997a07fe4bc81f6f6bd323801dc898371c2dbcda2082772e.jpg)

![1ee6b9e8c5328623ff8b7cf05536ab5de198090e7687ef10129dd0560eb75b50.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/1ee6b9e8c5328623ff8b7cf05536ab5de198090e7687ef10129dd0560eb75b50.jpg)

![20b18a85edd2813001ca0a2f17737373e1feb4badbad766011e71928afcd6a38.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/20b18a85edd2813001ca0a2f17737373e1feb4badbad766011e71928afcd6a38.jpg)

![2990ffa1453e5a67daa089ae00c525cf614de088b6551cbdb3db3c98c80bd58b.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/2990ffa1453e5a67daa089ae00c525cf614de088b6551cbdb3db3c98c80bd58b.jpg)

![47bde9433505a7444b72bc4a69d090f620e0f36f62c8ae18ec187a23819d3e89.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/47bde9433505a7444b72bc4a69d090f620e0f36f62c8ae18ec187a23819d3e89.jpg)

![49b17a0e6b03000800295a29fd8e256c2da53972c04e194c84efbd5b544212e3.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/49b17a0e6b03000800295a29fd8e256c2da53972c04e194c84efbd5b544212e3.jpg)

![5b8cb8ffec33bbb090e481ede447d7d969703945b5cb3bc220fa9830fa9947b5.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/5b8cb8ffec33bbb090e481ede447d7d969703945b5cb3bc220fa9830fa9947b5.jpg)

![5c7b8fbd003fdd14c40126a716ab10eaeb90661cdab17a2921e2ff47aa60604c.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/5c7b8fbd003fdd14c40126a716ab10eaeb90661cdab17a2921e2ff47aa60604c.jpg)

![5ed41d6840264ddcf4b21c52843f629a022b50d6e69052c715b24ae38b3af6ac.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/5ed41d6840264ddcf4b21c52843f629a022b50d6e69052c715b24ae38b3af6ac.jpg)

![6e2432ad12acfdc5334be22bdf44a086430f5afddbdd13bf3441621a551545bc.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/6e2432ad12acfdc5334be22bdf44a086430f5afddbdd13bf3441621a551545bc.jpg)

![7c9b8293276a2606569c05b197339a92aa60921b1d359b0c7a4a047bda35b34e.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/7c9b8293276a2606569c05b197339a92aa60921b1d359b0c7a4a047bda35b34e.jpg)

![a499f8db98c4db2c631fda0f2f8d1adc1198d3d8da527d17111d8af61a24e425.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/a499f8db98c4db2c631fda0f2f8d1adc1198d3d8da527d17111d8af61a24e425.jpg)

![abd941ffc680ebd2cfdb8f97f5e56cb1a9e7875515810505d9cee87138b43831.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/abd941ffc680ebd2cfdb8f97f5e56cb1a9e7875515810505d9cee87138b43831.jpg)

![bca6b31ba301db0a76604625a6e5186328c827e482070166a101632ac2527cbf.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/bca6b31ba301db0a76604625a6e5186328c827e482070166a101632ac2527cbf.jpg)

![bd1e14df7d28bea41c65cd16872fdf7a0bcf5fda33f0af01ba256e7cc20a494e.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/bd1e14df7d28bea41c65cd16872fdf7a0bcf5fda33f0af01ba256e7cc20a494e.jpg)

![cc2a72ecf5ba04a1c22d566826f558cf7da3b31d54e954bf2bcd4f0b1064d8eb.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/cc2a72ecf5ba04a1c22d566826f558cf7da3b31d54e954bf2bcd4f0b1064d8eb.jpg)

![d36316d9c9fa955128e49259448432ecbb558c774c11547ff9cfd42035e9e86f.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/d36316d9c9fa955128e49259448432ecbb558c774c11547ff9cfd42035e9e86f.jpg)

![df444c1fcdd84d5311f0e7c3884ec3221f018368dee89745bb1b81c669f05834.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/df444c1fcdd84d5311f0e7c3884ec3221f018368dee89745bb1b81c669f05834.jpg)

![eff08b89f0ee4ccc9a7de07f4333886fc0c4d7e36a919e2e4c9d4e03208c4a53.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/eff08b89f0ee4ccc9a7de07f4333886fc0c4d7e36a919e2e4c9d4e03208c4a53.jpg)

![f2586dca04006c7265bbdc01dee58cb345394bea76b3e2b446ac27e9743ca533.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/f2586dca04006c7265bbdc01dee58cb345394bea76b3e2b446ac27e9743ca533.jpg)

![f50bc1c90dfac68fe08e74abd9f0c2f2a1a7ba25bc9858018b4d801ab0953aa5.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/images/f50bc1c90dfac68fe08e74abd9f0c2f2a1a7ba25bc9858018b4d801ab0953aa5.jpg)

### Tables

![07726e504439fb72867a1a1c10b0cab76fe7bad4be54f212ffd3baca6828bc67.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/07726e504439fb72867a1a1c10b0cab76fe7bad4be54f212ffd3baca6828bc67.jpg)

![1336eae4128df3c3c0566129c1385eede6ef9201e43642e76e42557c6324fa7d.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/1336eae4128df3c3c0566129c1385eede6ef9201e43642e76e42557c6324fa7d.jpg)

![196beb43ba6e418a67efafae242bf12eb94002d1a19b0a154500df18c0896d2c.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/196beb43ba6e418a67efafae242bf12eb94002d1a19b0a154500df18c0896d2c.jpg)

![39e237c140a7fe9616de2af1ecf713241664d588cd76113875af18f0d7e5e269.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/39e237c140a7fe9616de2af1ecf713241664d588cd76113875af18f0d7e5e269.jpg)

![6c87217bd82cb13f1a6b31e681d75f6cef7f236517702800185fb412d40c757e.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/6c87217bd82cb13f1a6b31e681d75f6cef7f236517702800185fb412d40c757e.jpg)

![a520718dfe2f815a827493be9cbbe0bfe70aca58d33622d30f5583d314eac706.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/a520718dfe2f815a827493be9cbbe0bfe70aca58d33622d30f5583d314eac706.jpg)

![d7b08d172cb26aee28d5f49687aef2cb443aee2da7d37a084ff64e5e4e83c0a6.jpg](../icml_results/331_SoftJAX%20%26%20SoftTorch_%20Empowering%20Automatic%20Differentiation%20Libraries%20with%20Informative%20Gradients/tables/d7b08d172cb26aee28d5f49687aef2cb443aee2da7d37a084ff64e5e4e83c0a6.jpg)

<a id="paper-0327"></a>

## Safety Alignment of LMs via Non-cooperative Games


### Images

![56bcd799e1beca7706c7a0910bcc291c2fdcdcc43d8a42b91e9a04aefb7a2442.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/images/56bcd799e1beca7706c7a0910bcc291c2fdcdcc43d8a42b91e9a04aefb7a2442.jpg)

![697a92d6d0c14ebff0e48fca1bb2610ef780c08e8a16bd3f30b6ca5dff11b57f.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/images/697a92d6d0c14ebff0e48fca1bb2610ef780c08e8a16bd3f30b6ca5dff11b57f.jpg)

![7134cb50d6ef805c00cba689d2435fa0f5f194a02f45a53e4c49f8d127a73bb3.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/images/7134cb50d6ef805c00cba689d2435fa0f5f194a02f45a53e4c49f8d127a73bb3.jpg)

![745c1f04942aa1637ffd4a85422373f555237bbe9ded6bb9633aba808cabb174.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/images/745c1f04942aa1637ffd4a85422373f555237bbe9ded6bb9633aba808cabb174.jpg)

![7d5f3fb403efef8f728b52b073106bdc7d28521f0134345fac6763a82daafff9.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/images/7d5f3fb403efef8f728b52b073106bdc7d28521f0134345fac6763a82daafff9.jpg)

![a1904741d2572b72ef6cea937f8458dcebc65f5df0036acc146383ab356e04af.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/images/a1904741d2572b72ef6cea937f8458dcebc65f5df0036acc146383ab356e04af.jpg)

### Tables

![062d553a3b5bc886cdd0a8cc5318f4e0b1f37d9ef5a10a8acd1710663b3ff85b.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/062d553a3b5bc886cdd0a8cc5318f4e0b1f37d9ef5a10a8acd1710663b3ff85b.jpg)

![0b883a6d8f22e1693e118125611b70f08650fa72b9e6cd630fb99d4261e5724a.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/0b883a6d8f22e1693e118125611b70f08650fa72b9e6cd630fb99d4261e5724a.jpg)

![1d3d6d55abbddb2411b068af06820752c03f40f7d97ca9f822b4ba9a66c7ca0a.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/1d3d6d55abbddb2411b068af06820752c03f40f7d97ca9f822b4ba9a66c7ca0a.jpg)

![89e526ab9841ce5e957a258e0c77d18823454d63f42e05de2800a4579ec564f2.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/89e526ab9841ce5e957a258e0c77d18823454d63f42e05de2800a4579ec564f2.jpg)

![9d8ce27a5c79d55cd03dea32a846f63851bf1f24d3b8d8f60340bc4f7a13760b.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/9d8ce27a5c79d55cd03dea32a846f63851bf1f24d3b8d8f60340bc4f7a13760b.jpg)

![a54283234fddfb9b8ecc0b71b0b398635bd7a8757df7fa40d0de144e65dd762c.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/a54283234fddfb9b8ecc0b71b0b398635bd7a8757df7fa40d0de144e65dd762c.jpg)

![c54f3a43d55d38ec7faac7884390f28e35052b8fc54ed04865d90f0b5c03ad5c.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/c54f3a43d55d38ec7faac7884390f28e35052b8fc54ed04865d90f0b5c03ad5c.jpg)

![cda3785092c767fdd1f2f9dd8d198fa65c8e5c0df9f6cb11d4ff0f80e5681cb9.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/cda3785092c767fdd1f2f9dd8d198fa65c8e5c0df9f6cb11d4ff0f80e5681cb9.jpg)

![d3a7ef8c2790e16eaea7c6d0009230c889e5b37eae2c35744bffe4ed454e9504.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/d3a7ef8c2790e16eaea7c6d0009230c889e5b37eae2c35744bffe4ed454e9504.jpg)

![dffdee2aedbf3be3ea2e38e32ee33831fbb2297db84ea0c87261575b3a74897b.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/dffdee2aedbf3be3ea2e38e32ee33831fbb2297db84ea0c87261575b3a74897b.jpg)

![e7ce0ae3dbd583b1f6cfd94d03c37671c9d4326865b95399fee4a66392c4a568.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/e7ce0ae3dbd583b1f6cfd94d03c37671c9d4326865b95399fee4a66392c4a568.jpg)

![f0b3612116a8ebe4e3248f7a4c73af38d9273438edabed3b1e1d8866fc01db7f.jpg](../icml_results/332_Safety%20Alignment%20of%20LMs%20via%20Non-cooperative%20Games/tables/f0b3612116a8ebe4e3248f7a4c73af38d9273438edabed3b1e1d8866fc01db7f.jpg)

<a id="paper-0328"></a>

## Recurrent Equivariant Constraint Modulation: Learning Per-Layer Symmetry Relaxation from Data


### Images

![01f090351cf02cc7c30a010bbfb4e7384107e1efb2d99738a23d2bfaa67064a8.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/01f090351cf02cc7c30a010bbfb4e7384107e1efb2d99738a23d2bfaa67064a8.jpg)

![2db07ee40644688ab0d4e19a5e54d416264368d291e82dc2e1b4dc4930aa7089.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/2db07ee40644688ab0d4e19a5e54d416264368d291e82dc2e1b4dc4930aa7089.jpg)

![bdf4c9f42120f8a8e7efa33bb67a93784285e4962743ae549a9b963dc96284b0.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/bdf4c9f42120f8a8e7efa33bb67a93784285e4962743ae549a9b963dc96284b0.jpg)

![c067f0c5bd9801a946554947cc6787f0d75f0d38eb51b309667bc1d919086b93.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/c067f0c5bd9801a946554947cc6787f0d75f0d38eb51b309667bc1d919086b93.jpg)

![c5bf414345fad1220d84af7bf281971e2997445d204db3bd41678e8085b37490.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/c5bf414345fad1220d84af7bf281971e2997445d204db3bd41678e8085b37490.jpg)

![f5e07e88312464d7af2c74fd6862b7106d77033909562c81f9e8d5b12e3daaf8.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/f5e07e88312464d7af2c74fd6862b7106d77033909562c81f9e8d5b12e3daaf8.jpg)

![fb0c1bf8849068c2d73e84cde9a6ebf2afbc16be7946c5cfbe038ab74e39ced7.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/images/fb0c1bf8849068c2d73e84cde9a6ebf2afbc16be7946c5cfbe038ab74e39ced7.jpg)

### Tables

![3a14c877f79b5b4c53019e4c8570ed1ce97d7dc2aff95c17f7e753b193dff1b2.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/tables/3a14c877f79b5b4c53019e4c8570ed1ce97d7dc2aff95c17f7e753b193dff1b2.jpg)

![56d4cfaf2b44c6d1382984b9a12d5741903c25ea2153e42dbd89cba8187c6e75.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/tables/56d4cfaf2b44c6d1382984b9a12d5741903c25ea2153e42dbd89cba8187c6e75.jpg)

![59ed9dc7e1c437bcc14946ca7b4e973df310643c61983cf87e1a22fd7574f519.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/tables/59ed9dc7e1c437bcc14946ca7b4e973df310643c61983cf87e1a22fd7574f519.jpg)

![cc43ee0fbfdd31ac63d1a55e4ba28108a35a81383fdd4243e547e2b85020114c.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/tables/cc43ee0fbfdd31ac63d1a55e4ba28108a35a81383fdd4243e547e2b85020114c.jpg)

![d79559650fc4300792cccea90267c69c2cc59cf9038b2afc3076dda5ca3cb80b.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/tables/d79559650fc4300792cccea90267c69c2cc59cf9038b2afc3076dda5ca3cb80b.jpg)

![f6635ff9df42172a65f0629c9952f99dc2120f050f5f21d2eb403f0b2a5410c5.jpg](../icml_results/333_Recurrent%20Equivariant%20Constraint%20Modulation_%20Learning%20Per-Layer%20Symmetry%20Relaxation%20from%20Data/tables/f6635ff9df42172a65f0629c9952f99dc2120f050f5f21d2eb403f0b2a5410c5.jpg)

<a id="paper-0329"></a>

## Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment


### Images

![0310c84d30459e7081e0602dfd7ea2a375e6540ceef2aaf13416a57afc11b0b1.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/0310c84d30459e7081e0602dfd7ea2a375e6540ceef2aaf13416a57afc11b0b1.jpg)

![07b716a9adad5ed586bccec413268279840386834bcd833510d72bb177a9fad7.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/07b716a9adad5ed586bccec413268279840386834bcd833510d72bb177a9fad7.jpg)

![2c0a313879b76152953e3f836d32344d5269f3f20a46111a267b6cd7d50af05c.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/2c0a313879b76152953e3f836d32344d5269f3f20a46111a267b6cd7d50af05c.jpg)

![48d2cd2964f8a071d3d2c2bf128d32a2f7e011eea06505fc7fa4a895789b788e.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/48d2cd2964f8a071d3d2c2bf128d32a2f7e011eea06505fc7fa4a895789b788e.jpg)

![720281044f1f68f849c0fb53793dd5f0864926757e3e1102ee4ef4770c5e85bb.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/720281044f1f68f849c0fb53793dd5f0864926757e3e1102ee4ef4770c5e85bb.jpg)

![732eac931b7223cd52aa42b676a0c82c483e256312df1a73edf8f75793b906a1.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/732eac931b7223cd52aa42b676a0c82c483e256312df1a73edf8f75793b906a1.jpg)

![855770baec966919de176515411ccea3fa6b247eb616b57cfd7b12b577e715bd.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/855770baec966919de176515411ccea3fa6b247eb616b57cfd7b12b577e715bd.jpg)

![880d1c01c65536f71f9a15c33c1829b974cb8512eec5330a8214b737e2c95237.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/880d1c01c65536f71f9a15c33c1829b974cb8512eec5330a8214b737e2c95237.jpg)

![8ff413ae5e11b610f1040995013516f2db7e0402a254a862cd47821510865b61.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/8ff413ae5e11b610f1040995013516f2db7e0402a254a862cd47821510865b61.jpg)

![90539a5592269b181d6388970e82336dfa5748eb2021fe4ed4c5545eb5059b8c.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/90539a5592269b181d6388970e82336dfa5748eb2021fe4ed4c5545eb5059b8c.jpg)

![9604cc8f75fcce1b993f42789ccccc9828e607f97493d7194c5ca89dca4eec32.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/9604cc8f75fcce1b993f42789ccccc9828e607f97493d7194c5ca89dca4eec32.jpg)

![985272b6637a5755191f5ceaa588af16abbafaee80c7f0086f9f1e8485705d7b.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/985272b6637a5755191f5ceaa588af16abbafaee80c7f0086f9f1e8485705d7b.jpg)

![aed09abe7e8e8781f6f09064c745452ed4d871cd3fb6a4a149175d521e6ae242.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/aed09abe7e8e8781f6f09064c745452ed4d871cd3fb6a4a149175d521e6ae242.jpg)

![b049c8f80a0fe136012a70480b9f862c3c8481392ef0db7a2bfa884afcc46290.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/b049c8f80a0fe136012a70480b9f862c3c8481392ef0db7a2bfa884afcc46290.jpg)

![b7256a8144dfc838273ecef522a6804bade83f6954020977b24fb1be379af9b3.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/b7256a8144dfc838273ecef522a6804bade83f6954020977b24fb1be379af9b3.jpg)

![d7013487e39e2da9a1c4b9ea48fbebad98f169021d63dd3735e5e5840c2942a2.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/d7013487e39e2da9a1c4b9ea48fbebad98f169021d63dd3735e5e5840c2942a2.jpg)

![df8c2d76dbbe8f58d1f5c772e8c6e29a3834891650f34ee25a0bb306799d8c16.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/df8c2d76dbbe8f58d1f5c772e8c6e29a3834891650f34ee25a0bb306799d8c16.jpg)

![e5306731b48510f44063e2cd24c267c7651e3691b98f9bc5d3d74bbbc0a6b153.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/e5306731b48510f44063e2cd24c267c7651e3691b98f9bc5d3d74bbbc0a6b153.jpg)

![e9fb530dc13b1a862ff8178566b6bcce4cd1782d5c2d1ee81f2cf7a81a3736b4.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/e9fb530dc13b1a862ff8178566b6bcce4cd1782d5c2d1ee81f2cf7a81a3736b4.jpg)

![f6a74d8ff561c670f214df16728a79275b6f6a1c043678370aa8065b4bf1c608.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/f6a74d8ff561c670f214df16728a79275b6f6a1c043678370aa8065b4bf1c608.jpg)

![f78e51d96526eb9efbbbf07f0073683147cacc555eb8858384b71922d9684691.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/images/f78e51d96526eb9efbbbf07f0073683147cacc555eb8858384b71922d9684691.jpg)

### Tables

![3a4e2ac33bd67cdabcf0c600f76068d585bd7705a5c6d43b9b78ddadd7f09b8b.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/tables/3a4e2ac33bd67cdabcf0c600f76068d585bd7705a5c6d43b9b78ddadd7f09b8b.jpg)

![604c7ae07d76b0cedca0635ec1018fb6b5e9cc2604799744fbbe6a04d428a830.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/tables/604c7ae07d76b0cedca0635ec1018fb6b5e9cc2604799744fbbe6a04d428a830.jpg)

![7a562e032d8b88b9b542ddce98e7aaeb2a95e03076bce855cd7e688c68ec057e.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/tables/7a562e032d8b88b9b542ddce98e7aaeb2a95e03076bce855cd7e688c68ec057e.jpg)

![99b0c59bab7ce802c7cdff77f7dfd251b7306e5150361aa9c03198439ea91bf7.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/tables/99b0c59bab7ce802c7cdff77f7dfd251b7306e5150361aa9c03198439ea91bf7.jpg)

![f1a7bd052850c7440235589401aee646554910f1ceece01b3df97225fcd8262e.jpg](../icml_results/334_Offline%20Reinforcement%20Learning%20of%20High-Quality%20Behaviors%20Under%20Robust%20Style%20Alignment/tables/f1a7bd052850c7440235589401aee646554910f1ceece01b3df97225fcd8262e.jpg)

<a id="paper-0330"></a>

## BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs

### Images

![01f2d864392f03e52e1c8c6469220d29bd9afd869a6c1a58b48912cbcb1f4bae.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/01f2d864392f03e52e1c8c6469220d29bd9afd869a6c1a58b48912cbcb1f4bae.jpg)

![127279f53fb494cd791b3ce77a05e99002c07495ef140265d15d0ed158221d24.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/127279f53fb494cd791b3ce77a05e99002c07495ef140265d15d0ed158221d24.jpg)

![1a2ea5b8c8144695a2c41d6081c265039c97406ba008f104e6f32359df4ddb6e.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/1a2ea5b8c8144695a2c41d6081c265039c97406ba008f104e6f32359df4ddb6e.jpg)

![1fee55ff586bbae9100005b24a0a36ede9073e218d5ccf9f8b56a4c4a15753db.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/1fee55ff586bbae9100005b24a0a36ede9073e218d5ccf9f8b56a4c4a15753db.jpg)

![38d655722e610f925cb9247e68f063e3af3a8f5364654799a2757623143d20f6.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/38d655722e610f925cb9247e68f063e3af3a8f5364654799a2757623143d20f6.jpg)

![399000ab3c22ab636c39f6150e193a45a695219da31f57ccb9d6a4d6f2dcad96.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/399000ab3c22ab636c39f6150e193a45a695219da31f57ccb9d6a4d6f2dcad96.jpg)

![399d5d3eecf8f02baa880294fee41273eb3c34e1c9d033c6463760564e7dd9ca.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/399d5d3eecf8f02baa880294fee41273eb3c34e1c9d033c6463760564e7dd9ca.jpg)

![3a6735f5134735016b3eea76af8f2f69ad0ed7aad3d88c49ccf13657ce7c8722.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/3a6735f5134735016b3eea76af8f2f69ad0ed7aad3d88c49ccf13657ce7c8722.jpg)

![3e83366008b14418c5b3a7d883248667b990cc9472ff426442614fbc6200e163.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/3e83366008b14418c5b3a7d883248667b990cc9472ff426442614fbc6200e163.jpg)

![59e745c42252b15d266276d039a4f750cdc2de955264bc6f4ea2388dfee8d627.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/59e745c42252b15d266276d039a4f750cdc2de955264bc6f4ea2388dfee8d627.jpg)

![63de7f83b03490886afe15cd46da15f5109ad5423482c59f2fb3cc063685c701.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/63de7f83b03490886afe15cd46da15f5109ad5423482c59f2fb3cc063685c701.jpg)

![7c8bd8d8ecaf5c74e37a3ba55b5635f7771e2b939384f5683d3d62928960cbb1.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/7c8bd8d8ecaf5c74e37a3ba55b5635f7771e2b939384f5683d3d62928960cbb1.jpg)

![80fa62ea4a852b8d223889e637012c367f350675f84608e0cdfe2f678a651a7f.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/80fa62ea4a852b8d223889e637012c367f350675f84608e0cdfe2f678a651a7f.jpg)

![87b973b24786d0e3dc29ab830617a34a5d8400aa9179eff36207a817c7dffafd.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/87b973b24786d0e3dc29ab830617a34a5d8400aa9179eff36207a817c7dffafd.jpg)

![a0166eb387d6a1db9d7add5b6b878ceeded89327dcf4f3bf91f2368945c9f2b3.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/a0166eb387d6a1db9d7add5b6b878ceeded89327dcf4f3bf91f2368945c9f2b3.jpg)

![bd301934c88084405b6554e1db01dab43135e1a46ebd96aa5341aaa4078fe5ca.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/bd301934c88084405b6554e1db01dab43135e1a46ebd96aa5341aaa4078fe5ca.jpg)

![c19496224fdea79590cdeccf6fc97c9ecf098d0a39657d2128282b264b689ea9.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/c19496224fdea79590cdeccf6fc97c9ecf098d0a39657d2128282b264b689ea9.jpg)

![ca659dae383a5f8098b47f1b084e65a93c222491e302c03dc1e7018853150ab5.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/ca659dae383a5f8098b47f1b084e65a93c222491e302c03dc1e7018853150ab5.jpg)

![cacf57c09b3e1a2116375cdeda9900c9d9ff5030debe4f05fd3bed6b5822d5ee.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/cacf57c09b3e1a2116375cdeda9900c9d9ff5030debe4f05fd3bed6b5822d5ee.jpg)

![cf08b2f80fd55b2e317e2d00123e9e0ac296b9f27ee6722b02b5f2a01279b106.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/cf08b2f80fd55b2e317e2d00123e9e0ac296b9f27ee6722b02b5f2a01279b106.jpg)

![d1fdbacfac7f34add72ee5a6bc44526422dc6bae9807312306e7e1ac4e6a5f62.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/d1fdbacfac7f34add72ee5a6bc44526422dc6bae9807312306e7e1ac4e6a5f62.jpg)

![df62509e32681e2fa193cae3cb66be0e520a7304822ddf98dd1124b0520bf20a.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/df62509e32681e2fa193cae3cb66be0e520a7304822ddf98dd1124b0520bf20a.jpg)

![df821f7a830aa0d1e0d965d43269dcb419ecdb93c6c424d84eb116969352fcba.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/df821f7a830aa0d1e0d965d43269dcb419ecdb93c6c424d84eb116969352fcba.jpg)

![f2860d56b9f9cb7d4bb06d4bc66125879a54ad8a1e69aca69753ee7c68ec06be.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/images/f2860d56b9f9cb7d4bb06d4bc66125879a54ad8a1e69aca69753ee7c68ec06be.jpg)

### Tables

![00dcdd3ce6a74129f7943a772d9b0dfdf15866af3edb558c03e4d0e0932a995f.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/00dcdd3ce6a74129f7943a772d9b0dfdf15866af3edb558c03e4d0e0932a995f.jpg)

![108aa01417d820719be93c2ad447f65f74fba05eea27ea0c992e1bf3c19393c5.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/108aa01417d820719be93c2ad447f65f74fba05eea27ea0c992e1bf3c19393c5.jpg)

![3cfd4aec3433e3d6d8262d2d721c146f82e31906421a7fdf98ae29d99600e92d.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/3cfd4aec3433e3d6d8262d2d721c146f82e31906421a7fdf98ae29d99600e92d.jpg)

![568693a7a669ba277a983e0b25c300880ebbad9816ed058ab9f8c1ff507489ad.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/568693a7a669ba277a983e0b25c300880ebbad9816ed058ab9f8c1ff507489ad.jpg)

![87247cd582ed1d70d2d48fb6d137cbb6f3c814035ca1da27e33e2e7e0aa36061.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/87247cd582ed1d70d2d48fb6d137cbb6f3c814035ca1da27e33e2e7e0aa36061.jpg)

![de568cbbd8ff366707370cd2a9904a930969ddea2e0ebf81b91871a4e7fb750c.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/de568cbbd8ff366707370cd2a9904a930969ddea2e0ebf81b91871a4e7fb750c.jpg)

![f49ca0f9bbf93d02fc64967c3bac08f764651dff5ec00cbb2ad7db60a9398fa0.jpg](../icml_results/335_BrokenMath_%20A%20Benchmark%20for%20Sycophancy%20in%20Theorem%20Proving%20with%20LLMs/tables/f49ca0f9bbf93d02fc64967c3bac08f764651dff5ec00cbb2ad7db60a9398fa0.jpg)
