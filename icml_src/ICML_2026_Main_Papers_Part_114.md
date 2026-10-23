# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 114 of 197

## 目录 (Table of Contents)

1. [Group Distributionally Robust Optimization-Driven RL for LLM Reasoning](#paper-3730)
2. [CausalProfiler: Generating Synthetic Benchmarks for Rigorous and Transparent Evaluation of Causal Machine Learning](#paper-3731)
3. [Memory Savings at What Cost? A Study of Alternatives to Backpropagation](#paper-3732)
4. [Mosaic: Runtime-Efficient Multi-Agent Embodied Planning](#paper-3733)
5. [Interpretability Driven Evolutionary Approach for the Design of Biological Sequences](#paper-3734)
6. [Continual Segmentation under Joint Nonstationarity](#paper-3735)
7. [Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation](#paper-3736)
8. [Decouple and Cache: KV Cache Construction for Streaming Video Understanding](#paper-3737)
9. [Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](#paper-3738)
10. [Unveiling the Visual Counting Bottleneck in Vision-Language Models](#paper-3739)
11. [Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](#paper-3740)
12. [EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning](#paper-3741)
13. [Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs](#paper-3742)
14. [Multi-Agent Teams Hold Experts Back](#paper-3743)
15. [STARCaster: Spatio-Temporal AutoRegressive Video Diffusion for Identity- and View-Aware Talking Portraits](#paper-3744)
16. [Learnability-Informed Fine-Tuning of Diffusion Language Models](#paper-3745)
17. [Online Social Welfare Function-based Resource Allocation](#paper-3746)
18. [TabularBERT: Binning-Based Self-Supervised Learning for Tabular Representation](#paper-3747)
19. [Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach](#paper-3748)
20. [Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling](#paper-3749)
21. [Convergent World Representations and Divergent Tasks](#paper-3750)
22. [Sharpness-Aware Minimization Can Hallucinate Minimizers](#paper-3751)
23. [Q-Delta: Beyond Key–Value Associative State Evolution](#paper-3752)
24. [Class-Grouped Normalized Momentum and Faster Hyperparameter Exploration to Tackle Class Imbalance in Federated Learning](#paper-3753)
25. [$\texttt{Multi}^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](#paper-3754)
26. [Euler–Poincaré Neural Dynamics: A Geometric-Mechanics Framework for Scientific Simulation](#paper-3755)
27. [Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces](#paper-3756)
28. [Saliency-Aware Model Merging](#paper-3757)
29. [V-LynX: Token Interface Alignment for Video+X LLMs](#paper-3758)
30. [SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures](#paper-3759)
31. [Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion](#paper-3760)
32. [The Information Geometry of Softmax: Probing and Steering](#paper-3761)
33. [Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL](#paper-3762)

---


<a id="paper-3730"></a>

## Group Distributionally Robust Optimization-Driven RL for LLM Reasoning

### Images

![11b680e7dca05d9981e774e0e5658fe6ce230db6d3dbc2b4a9aa6d4876238e16.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/11b680e7dca05d9981e774e0e5658fe6ce230db6d3dbc2b4a9aa6d4876238e16.jpg)

![12820af21c99f5b04618c4231c9d322fe28120079f89b1f3f01924052a2ccbb1.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/12820af21c99f5b04618c4231c9d322fe28120079f89b1f3f01924052a2ccbb1.jpg)

![2dfedfbd49946ad8854a4e5bcfe660cb09dc436e2a44305bbc1c14368f46de63.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/2dfedfbd49946ad8854a4e5bcfe660cb09dc436e2a44305bbc1c14368f46de63.jpg)

![334132bfe2006fa0178e04021177bd8e06ac61d11ac15db94908fb510c552437.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/334132bfe2006fa0178e04021177bd8e06ac61d11ac15db94908fb510c552437.jpg)

![3533e9510164d5c861f3caaeb2eb14a65afa247bc9a19c81d271bdad667652e0.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/3533e9510164d5c861f3caaeb2eb14a65afa247bc9a19c81d271bdad667652e0.jpg)

![5355c2456bcf9019310d2e32a9d1b1fe1a9fae01643bb35a9d902fbde27a7946.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/5355c2456bcf9019310d2e32a9d1b1fe1a9fae01643bb35a9d902fbde27a7946.jpg)

![5a19a925bd804cc050ffe2d0d1abb933c84950c294b346e4f9137db961c94d8a.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/5a19a925bd804cc050ffe2d0d1abb933c84950c294b346e4f9137db961c94d8a.jpg)

![5dda6551ea30467bc87558d3f1eaad6d1094a2d54a1e98859db233e683f9c921.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/5dda6551ea30467bc87558d3f1eaad6d1094a2d54a1e98859db233e683f9c921.jpg)

![c0d3918db461446f7571b5d83760ad589bc655747cdccbce56e6154f77507d3c.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/c0d3918db461446f7571b5d83760ad589bc655747cdccbce56e6154f77507d3c.jpg)

![d0592b6eac86719056976ce3930599ad7b318d3c18d4e4cecb3527fc67232b98.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/d0592b6eac86719056976ce3930599ad7b318d3c18d4e4cecb3527fc67232b98.jpg)

![d364c2c4d5cadc08395f669610f1074c8b7e73da00aa000e958c116923092091.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/d364c2c4d5cadc08395f669610f1074c8b7e73da00aa000e958c116923092091.jpg)

![e5ff4e186bc9808c4a692b31b50fa1b4009c48854c8f1ab3b3553d730e83dd1a.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/images/e5ff4e186bc9808c4a692b31b50fa1b4009c48854c8f1ab3b3553d730e83dd1a.jpg)

### Tables

![05910b3efee3a677ec4de965497af3dd85fa92dca68192be64fbadc141309e13.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/05910b3efee3a677ec4de965497af3dd85fa92dca68192be64fbadc141309e13.jpg)

![1f731284cb9b52524a881365a323fb189896b12db9ae4884d6b0827db349c97e.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/1f731284cb9b52524a881365a323fb189896b12db9ae4884d6b0827db349c97e.jpg)

![1fb21ced81a0b4125488008657e008fd5cef5e6a7991fe6d8c988e58816bdbef.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/1fb21ced81a0b4125488008657e008fd5cef5e6a7991fe6d8c988e58816bdbef.jpg)

![248845af73f48cb753a3bbcf5ad0be3824a3e1e1281c7e14035cbd0626adecb4.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/248845af73f48cb753a3bbcf5ad0be3824a3e1e1281c7e14035cbd0626adecb4.jpg)

![27c6817d18c8191560faa65a02232b3a8c4e84edb4cee8967698b38d8cc211c0.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/27c6817d18c8191560faa65a02232b3a8c4e84edb4cee8967698b38d8cc211c0.jpg)

![32bfe5009d7d474748bb534dd4a66648c717db527956ba08a4a4fc0f2e9138a0.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/32bfe5009d7d474748bb534dd4a66648c717db527956ba08a4a4fc0f2e9138a0.jpg)

![54f9c0eb758f834a91da517bf1a5c58033894c5a42a0d4b335d75f5dc01cff4e.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/54f9c0eb758f834a91da517bf1a5c58033894c5a42a0d4b335d75f5dc01cff4e.jpg)

![6e35fbd26c12e28832c3d182313ba2418b8f9a748393f5f9a5ce898ad738c58c.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/6e35fbd26c12e28832c3d182313ba2418b8f9a748393f5f9a5ce898ad738c58c.jpg)

![73812515953af23d58bfcae2f47e12cac5c963cf0c41ee0bc044434fcc6ec33d.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/73812515953af23d58bfcae2f47e12cac5c963cf0c41ee0bc044434fcc6ec33d.jpg)

![9478d012a1d70b82a19c9510c7ae89a23664009bfdbf1fbcd9595558b14698e6.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/9478d012a1d70b82a19c9510c7ae89a23664009bfdbf1fbcd9595558b14698e6.jpg)

![c55b5a51f1fe3a9597a01a714dfdc61baa8ce588983c31fa9e0b0b41f827be34.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/c55b5a51f1fe3a9597a01a714dfdc61baa8ce588983c31fa9e0b0b41f827be34.jpg)

![d7983fa5f676375bc9071372f19e3af329d9c09bc91cbee839e66eb6c4f22bc1.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/d7983fa5f676375bc9071372f19e3af329d9c09bc91cbee839e66eb6c4f22bc1.jpg)

![da90c31cdea07261286dd64d0c2e4a263ea3abf88c37eb3560b248cf3bafe00c.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/da90c31cdea07261286dd64d0c2e4a263ea3abf88c37eb3560b248cf3bafe00c.jpg)

![dcf347e6693d64eb5447f00119313e77325895bda13726a1a77f2a611b546b70.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/dcf347e6693d64eb5447f00119313e77325895bda13726a1a77f2a611b546b70.jpg)

![dd0e2e5c49ddc1a11a3f003587f3181316d9a5110015ce49a40d9801e62a33f2.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/dd0e2e5c49ddc1a11a3f003587f3181316d9a5110015ce49a40d9801e62a33f2.jpg)

![f1f8e9eedd05f861bc8643c5e8633cdc9cc0773e0ff210bae3d282c7bbfada74.jpg](../icml_results/3769_AdvEvo-MARL_%20Shaping%20Internalized%20Safety%20through%20Adversarial%20Co-Evolution%20in%20Multi-Agent%20Reinforceme/tables/f1f8e9eedd05f861bc8643c5e8633cdc9cc0773e0ff210bae3d282c7bbfada74.jpg)

## Group Distributionally Robust Optimization-Driven RL for LLM Reasoning


### Images

![0dbbd408207463fb295104967343dfc9ecb0c202ca97329ada6a97108b0f100b.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/0dbbd408207463fb295104967343dfc9ecb0c202ca97329ada6a97108b0f100b.jpg)

![201aa9d217829f83bf0f6b54b5bc9294cd4ab7fae44cef029cd5e87eb3906bb7.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/201aa9d217829f83bf0f6b54b5bc9294cd4ab7fae44cef029cd5e87eb3906bb7.jpg)

![6d275cb9c90c84820f6cf173dbb9d48f54f48d99289de0c7ac1afd8f80ba5b5e.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/6d275cb9c90c84820f6cf173dbb9d48f54f48d99289de0c7ac1afd8f80ba5b5e.jpg)

![7646d1d5e82ae655cc97421212b7ecc6e0793f78cf852c3a08cb6d27a65ea73e.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/7646d1d5e82ae655cc97421212b7ecc6e0793f78cf852c3a08cb6d27a65ea73e.jpg)

![9cbc984c43aa26cbbd2572c21c2e328d5bf3ec95de1c36d2e082c3c0e95f00de.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/9cbc984c43aa26cbbd2572c21c2e328d5bf3ec95de1c36d2e082c3c0e95f00de.jpg)

![b9609b2a6464fb56c81c84a9fd0126552948d7ab8a7fea15fe91b1c7649c5ee1.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/b9609b2a6464fb56c81c84a9fd0126552948d7ab8a7fea15fe91b1c7649c5ee1.jpg)

![dc66c56e4c245270913904ffe3d9039b674abd2f28519e833e83d2ed2ca610f5.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/dc66c56e4c245270913904ffe3d9039b674abd2f28519e833e83d2ed2ca610f5.jpg)

![ef765c75fc91adc96b491b23c7c98d19dd5cbc56f4da6c5293fd0d5c6951a5bf.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/images/ef765c75fc91adc96b491b23c7c98d19dd5cbc56f4da6c5293fd0d5c6951a5bf.jpg)

### Tables

![c7090a328618d75fccb45b6f9e4515f792e177db7f5e5861d4477fde7c5558b5.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/tables/c7090a328618d75fccb45b6f9e4515f792e177db7f5e5861d4477fde7c5558b5.jpg)

![d083cae829f84599586de3f4764a774187528a07fcd88e57cfeea9561e7a7e3a.jpg](../icml_results/3770_Group%20Distributionally%20Robust%20Optimization-Driven%20RL%20for%20LLM%20Reasoning/tables/d083cae829f84599586de3f4764a774187528a07fcd88e57cfeea9561e7a7e3a.jpg)

<a id="paper-3731"></a>

## CausalProfiler: Generating Synthetic Benchmarks for Rigorous and Transparent Evaluation of Causal Machine Learning


### Images

![001049c5aa0a1d55eddb70d73baee49a3e733ec28dcd51be3b6ebefd0c79234c.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/001049c5aa0a1d55eddb70d73baee49a3e733ec28dcd51be3b6ebefd0c79234c.jpg)

![004a9828da5d95665285aae388fe30d06604b31867e69ec55e56dc6c0d3e771c.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/004a9828da5d95665285aae388fe30d06604b31867e69ec55e56dc6c0d3e771c.jpg)

![18bc989b17f5f29ad62f4c2e2087764387e96c4f2b2980199f485d9d7d69cc96.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/18bc989b17f5f29ad62f4c2e2087764387e96c4f2b2980199f485d9d7d69cc96.jpg)

![2c78406c7b6c0c9495c3969c2e7ed9a1d8f0e3cc9c872ee9e9550fa9f202d89c.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/2c78406c7b6c0c9495c3969c2e7ed9a1d8f0e3cc9c872ee9e9550fa9f202d89c.jpg)

![2f5b58d3aa7f8a8cca25094fcbc23a2bfbefa76ac5fc2a478e2223b5626d68bd.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/2f5b58d3aa7f8a8cca25094fcbc23a2bfbefa76ac5fc2a478e2223b5626d68bd.jpg)

![3edf5b97abd8463389ed2d23b1b025fe02846ee95959823e88ee7c85162cc8e9.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/3edf5b97abd8463389ed2d23b1b025fe02846ee95959823e88ee7c85162cc8e9.jpg)

![40723b4078f24b7100ca04d66f96d5ccac67848e96cc8fb5835d188457409bf6.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/40723b4078f24b7100ca04d66f96d5ccac67848e96cc8fb5835d188457409bf6.jpg)

![4ba6c6796e03082ab30af31a9dea67d088a0ee4211f0710e1ff301a9028222c1.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/4ba6c6796e03082ab30af31a9dea67d088a0ee4211f0710e1ff301a9028222c1.jpg)

![66225b6d674962fc4496308b635a138945906df9b208b73783597c3d458b0374.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/66225b6d674962fc4496308b635a138945906df9b208b73783597c3d458b0374.jpg)

![7e268e029eb767f971d460771f0fc29dc8e5e000e69ec28a96014629f57fe80a.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/7e268e029eb767f971d460771f0fc29dc8e5e000e69ec28a96014629f57fe80a.jpg)

![8a276c33c6f778ede2f71e50474f8024b4ac0c9c29e41fac5bdb86b4bdc264de.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/8a276c33c6f778ede2f71e50474f8024b4ac0c9c29e41fac5bdb86b4bdc264de.jpg)

![956e680db8546a2e5205c3edb1859b09ab70ba9b624502312789a1568eafcc17.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/956e680db8546a2e5205c3edb1859b09ab70ba9b624502312789a1568eafcc17.jpg)

![a6bdc6cabb4965fb39caefea7a231b97b44656f87649e9c41b41ce444ab946a8.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/a6bdc6cabb4965fb39caefea7a231b97b44656f87649e9c41b41ce444ab946a8.jpg)

![afbd5fc6ce349664661b539433b1988aa021ecbf13524d1d6d6aef55934763fb.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/afbd5fc6ce349664661b539433b1988aa021ecbf13524d1d6d6aef55934763fb.jpg)

![e44a25b56087c2b5f135be7e28628dfb9045f879807269261a6ff07b9c439033.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/e44a25b56087c2b5f135be7e28628dfb9045f879807269261a6ff07b9c439033.jpg)

![fe5805fb394bc101516c1079d774672afa151bdf2a69444768a7f14f67f2f9a4.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/images/fe5805fb394bc101516c1079d774672afa151bdf2a69444768a7f14f67f2f9a4.jpg)

### Tables

![030d94f31023a736f73933b5a7696f7bd247affc6a916b4401e0f8be344065c1.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/030d94f31023a736f73933b5a7696f7bd247affc6a916b4401e0f8be344065c1.jpg)

![0d1e41a9514eda7e2d8d193aa61a9807f3b8715461877d8475f0e2437bed1653.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/0d1e41a9514eda7e2d8d193aa61a9807f3b8715461877d8475f0e2437bed1653.jpg)

![1285f1763dcef7a9a025ff6d906ce4dead7eca738fe46e4bc144a7197e3b3a13.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/1285f1763dcef7a9a025ff6d906ce4dead7eca738fe46e4bc144a7197e3b3a13.jpg)

![16e1f15b5f62ea63517f8f63d0ff81da5ecf7ac88d95fb77a73c71029d8b279e.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/16e1f15b5f62ea63517f8f63d0ff81da5ecf7ac88d95fb77a73c71029d8b279e.jpg)

![1756b4707226b8d24d7ecdf482a9273c5f1718bf66f5da343bdc80f988828e83.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/1756b4707226b8d24d7ecdf482a9273c5f1718bf66f5da343bdc80f988828e83.jpg)

![1c6fe4d93bcadc321082930e2683f7e391cf37d7a047cd168e4323c88b18f595.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/1c6fe4d93bcadc321082930e2683f7e391cf37d7a047cd168e4323c88b18f595.jpg)

![1efba25afe31de66f458d7c16ea57ae7881ef72a5aab5349213622fab74dd6cf.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/1efba25afe31de66f458d7c16ea57ae7881ef72a5aab5349213622fab74dd6cf.jpg)

![20de826f2efa8b75ec764e3aab97eff70fbf26f23435e8c9eab96eff28ab5097.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/20de826f2efa8b75ec764e3aab97eff70fbf26f23435e8c9eab96eff28ab5097.jpg)

![247b7a8c5af9d899695808dcb93a883b71faa0065fa2840c327265e6fe2805ab.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/247b7a8c5af9d899695808dcb93a883b71faa0065fa2840c327265e6fe2805ab.jpg)

![299305819ff9c42fae5a0a965c10f7f5439090432bd52798e9f5a7d1743db140.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/299305819ff9c42fae5a0a965c10f7f5439090432bd52798e9f5a7d1743db140.jpg)

![3d9c00f65d0544bfa3bb1cea6b4e47e5c0a3a2c9d9990ced2ebf00d150ac109c.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/3d9c00f65d0544bfa3bb1cea6b4e47e5c0a3a2c9d9990ced2ebf00d150ac109c.jpg)

![4847aa396a5ac77ee39e40e70364a8015bb80b3e900e6fd68d39b5a53877fd3a.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/4847aa396a5ac77ee39e40e70364a8015bb80b3e900e6fd68d39b5a53877fd3a.jpg)

![4c18b449c08a2d80e9cd034625e2dcdc86cbc699b27989673403f0971973e9e1.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/4c18b449c08a2d80e9cd034625e2dcdc86cbc699b27989673403f0971973e9e1.jpg)

![542ce8e9db1626bc1a002d76c8f979b76f4d2b3bb90a103d2f52ca735c6d6430.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/542ce8e9db1626bc1a002d76c8f979b76f4d2b3bb90a103d2f52ca735c6d6430.jpg)

![67bbcb04fcf1840936644d95078f410beb3bb3f7fa0d6e4bd8749366c806129c.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/67bbcb04fcf1840936644d95078f410beb3bb3f7fa0d6e4bd8749366c806129c.jpg)

![7557cb526841af80c5e17ee1f40f71c0ca0f8b1079ee9bc7b9fc5e017a2d7195.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/7557cb526841af80c5e17ee1f40f71c0ca0f8b1079ee9bc7b9fc5e017a2d7195.jpg)

![804d593fa08f61218771dfe3553382b856fae094a6be4587bc03f044814493c1.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/804d593fa08f61218771dfe3553382b856fae094a6be4587bc03f044814493c1.jpg)

![8b2582ebc2e7199dec7ccb685d25842c52b79d35ad074dada0e3db58b019a728.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/8b2582ebc2e7199dec7ccb685d25842c52b79d35ad074dada0e3db58b019a728.jpg)

![90cab01b02d5959a804327dbc8de419d71da136e1997dbccd26a674a42480484.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/90cab01b02d5959a804327dbc8de419d71da136e1997dbccd26a674a42480484.jpg)

![9bc6a4b671dfa65065731bd78ce26e2cb9df654202dbd15b711472cfdeb29398.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/9bc6a4b671dfa65065731bd78ce26e2cb9df654202dbd15b711472cfdeb29398.jpg)

![a1f6866abcae8e31a417233716ec59196ba774dfdf8b14dce02a3ed082334507.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/a1f6866abcae8e31a417233716ec59196ba774dfdf8b14dce02a3ed082334507.jpg)

![a66e6d0acaef9074047c3ddb4cbaf6fcaeb07460d49c9a1525e0ef54f4da852c.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/a66e6d0acaef9074047c3ddb4cbaf6fcaeb07460d49c9a1525e0ef54f4da852c.jpg)

![abadbe0e9efc65f4308a509b17d505e763206faf958b8590aff3a903cca4b4a1.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/abadbe0e9efc65f4308a509b17d505e763206faf958b8590aff3a903cca4b4a1.jpg)

![b4d28c60ca32df3f59529cad44ab6db205760ad9630243e0fbe2bb12618632b0.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/b4d28c60ca32df3f59529cad44ab6db205760ad9630243e0fbe2bb12618632b0.jpg)

![b511317a38bb4b5a296a4e1059c54df90d04db9b439313e4f44f2c89723e616f.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/b511317a38bb4b5a296a4e1059c54df90d04db9b439313e4f44f2c89723e616f.jpg)

![b5b64b3bf1b6f95db5906e78d567494fb04e639c649c00a4e4c0d1298b6093db.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/b5b64b3bf1b6f95db5906e78d567494fb04e639c649c00a4e4c0d1298b6093db.jpg)

![b758a557c910225570fbe71295a50899e5b33a6c9d58c0f4264f39ec71b8f2c2.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/b758a557c910225570fbe71295a50899e5b33a6c9d58c0f4264f39ec71b8f2c2.jpg)

![c13b7a1a1e2f9227862ba4b62c1c655f7af910ea989b6001577ac8a06c221033.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/c13b7a1a1e2f9227862ba4b62c1c655f7af910ea989b6001577ac8a06c221033.jpg)

![cc63244d01a6def773ebf648e1e26fcb9ce67fe048ce723c97a3f8981a80c83a.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/cc63244d01a6def773ebf648e1e26fcb9ce67fe048ce723c97a3f8981a80c83a.jpg)

![d14b46c64d43ffbdf35c67cf7d8f2d2fb4f58e1810ee272b848496eb9b6969f6.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/d14b46c64d43ffbdf35c67cf7d8f2d2fb4f58e1810ee272b848496eb9b6969f6.jpg)

![e178a0a018c01a4a5158f3f9380555c2c254edc53f63424499e70d40f3a3b184.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/e178a0a018c01a4a5158f3f9380555c2c254edc53f63424499e70d40f3a3b184.jpg)

![e2f0e0b20e7f5d7d791bc09d926336641eee8f98b7ff3bfa4e643fbe1a372655.jpg](../icml_results/3771_CausalProfiler_%20Generating%20Synthetic%20Benchmarks%20for%20Rigorous%20and%20Transparent%20Evaluation%20of%20Causal%20Ma/tables/e2f0e0b20e7f5d7d791bc09d926336641eee8f98b7ff3bfa4e643fbe1a372655.jpg)

<a id="paper-3732"></a>

## Memory Savings at What Cost? A Study of Alternatives to Backpropagation


### Images

![010c813ba70a271a3ae699fb35087a24a2604838edfd6ea29eec9a5aa2ca62c3.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/010c813ba70a271a3ae699fb35087a24a2604838edfd6ea29eec9a5aa2ca62c3.jpg)

![13cbee91e0e442ef9be5fc9be13e8e172c0150f08bf28a625eae2a35a1323c50.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/13cbee91e0e442ef9be5fc9be13e8e172c0150f08bf28a625eae2a35a1323c50.jpg)

![151efceb8153be5f1e06803f15598c3f5e3b68f35bb9785b1e62bba6c64a9b7b.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/151efceb8153be5f1e06803f15598c3f5e3b68f35bb9785b1e62bba6c64a9b7b.jpg)

![1ecf44b3ce8a6538a0b342c42b0a2cee46051612ed1c7237bfaf868d15ac9559.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/1ecf44b3ce8a6538a0b342c42b0a2cee46051612ed1c7237bfaf868d15ac9559.jpg)

![23852f157f39744d242a965874dfb98ab0bb12ce090125273a944bdadc1ff96a.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/23852f157f39744d242a965874dfb98ab0bb12ce090125273a944bdadc1ff96a.jpg)

![39e1fcf283448542e71744fc0825abbd3709205488df8437e2de496355522e1b.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/39e1fcf283448542e71744fc0825abbd3709205488df8437e2de496355522e1b.jpg)

![514e21d4189516b516a84dad6113bb9b9be41d101228f2456b1a09f55f68349e.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/514e21d4189516b516a84dad6113bb9b9be41d101228f2456b1a09f55f68349e.jpg)

![608d3bbecef050ecacb9ff72fd1dcc87ebc469aebe511ae82a64fd54f53406cc.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/608d3bbecef050ecacb9ff72fd1dcc87ebc469aebe511ae82a64fd54f53406cc.jpg)

![6dcca2a89dbd54fdbd373357e2be6a72f9275e617a18f4e60a13b0180f8f0e79.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/6dcca2a89dbd54fdbd373357e2be6a72f9275e617a18f4e60a13b0180f8f0e79.jpg)

![7e59801b99b9dc87be55715337aa051b2e1824d5782aba44b3770b09cb748ffd.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/7e59801b99b9dc87be55715337aa051b2e1824d5782aba44b3770b09cb748ffd.jpg)

![93aa5cb9703b12c6ee5c7b71caf07e8ba28c7d5b3662404ac5eeeab3093197e2.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/93aa5cb9703b12c6ee5c7b71caf07e8ba28c7d5b3662404ac5eeeab3093197e2.jpg)

![a25503b8bac7d8468993c0cb27f3483338aade43b4213a0536d18ced1ca68477.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/a25503b8bac7d8468993c0cb27f3483338aade43b4213a0536d18ced1ca68477.jpg)

![ae94582ba59006312d8f260e4c35db24065d1c9a44bb5109f8883190e6628842.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/ae94582ba59006312d8f260e4c35db24065d1c9a44bb5109f8883190e6628842.jpg)

![b9ff3ca4f8cace0e489073332d0e21bf84886c7b6f198954e13233202b899e2e.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/b9ff3ca4f8cace0e489073332d0e21bf84886c7b6f198954e13233202b899e2e.jpg)

![e3a7a72427f828f04170638e52277bf391ccb367abaf9d368ced26b6ee3d6d8a.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/e3a7a72427f828f04170638e52277bf391ccb367abaf9d368ced26b6ee3d6d8a.jpg)

![f02fb08e2b0b2b1d37e0c4e2d5dfcc135e906f16012111ed6f256168ea33e4f6.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/images/f02fb08e2b0b2b1d37e0c4e2d5dfcc135e906f16012111ed6f256168ea33e4f6.jpg)

### Tables

![297375762a2efcbcd7152bfdb815884a93979f6741c6910204e354a3cf16eb41.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/297375762a2efcbcd7152bfdb815884a93979f6741c6910204e354a3cf16eb41.jpg)

![2d9f5c63e8eb2549c0c87e86b35b2ef689747ec2555315c40fa7eed416e14e43.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/2d9f5c63e8eb2549c0c87e86b35b2ef689747ec2555315c40fa7eed416e14e43.jpg)

![3246a71c557503909c3eeab1df479ffa88854d5dff8eec57d1254793651f1cec.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/3246a71c557503909c3eeab1df479ffa88854d5dff8eec57d1254793651f1cec.jpg)

![3da37496011e37d8013c1c587b26e3984220a2c9a29dcdab54ba97234ab86421.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/3da37496011e37d8013c1c587b26e3984220a2c9a29dcdab54ba97234ab86421.jpg)

![4c0bafcdafc483892ff88ad3f2da70158bbb6ecfa21eca62248e1298064fd0f1.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/4c0bafcdafc483892ff88ad3f2da70158bbb6ecfa21eca62248e1298064fd0f1.jpg)

![4e48e8edaffb8aff90d35971431d1e71cd3d86cd0398440259da96002702ff6e.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/4e48e8edaffb8aff90d35971431d1e71cd3d86cd0398440259da96002702ff6e.jpg)

![64e13e063ff9ed58e8ec98954cc658b33950e2cbd239e95584401616b5b50839.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/64e13e063ff9ed58e8ec98954cc658b33950e2cbd239e95584401616b5b50839.jpg)

![65e475503552480ab6b8cb0723f1f73a0490968d9cc5a13629a01e06e8b97e54.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/65e475503552480ab6b8cb0723f1f73a0490968d9cc5a13629a01e06e8b97e54.jpg)

![709acc04e7e39e143641ae4a9e9252450a5ad5e760df23b411a4b159eb96d8b4.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/709acc04e7e39e143641ae4a9e9252450a5ad5e760df23b411a4b159eb96d8b4.jpg)

![73681424d730158c047f96e615c7a0f0027eb0ba65ebaf2d86e3c0c0d5e76d6e.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/73681424d730158c047f96e615c7a0f0027eb0ba65ebaf2d86e3c0c0d5e76d6e.jpg)

![8c18c7d14d44eb92a27e24ab7e8f203400ec7cbd47337fc6f3ee56536919ec3b.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/8c18c7d14d44eb92a27e24ab7e8f203400ec7cbd47337fc6f3ee56536919ec3b.jpg)

![9c630fdf544a9025bd2ab556c18988778240a9a388ec69b2a562c1ced3c405f3.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/9c630fdf544a9025bd2ab556c18988778240a9a388ec69b2a562c1ced3c405f3.jpg)

![9e34e83341768020870a578130d6980aa165f3c0231e7d4f3ff49adc2cceec3a.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/9e34e83341768020870a578130d6980aa165f3c0231e7d4f3ff49adc2cceec3a.jpg)

![b0564edc90f08e0c65603cc70b857f3ee0e701e94a027c4753bf8ce350f6d574.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/b0564edc90f08e0c65603cc70b857f3ee0e701e94a027c4753bf8ce350f6d574.jpg)

![f3f7a66e7340236e3be5c2fc48ffc1ec684299dcfd16b98071562b2ebece15c8.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/f3f7a66e7340236e3be5c2fc48ffc1ec684299dcfd16b98071562b2ebece15c8.jpg)

![f97bd854149980b85e5e44f48a7cd004250be64ff81165fa410c4eb7bce90c68.jpg](../icml_results/3772_Memory%20Savings%20at%20What%20Cost_%20A%20Study%20of%20Alternatives%20to%20Backpropagation/tables/f97bd854149980b85e5e44f48a7cd004250be64ff81165fa410c4eb7bce90c68.jpg)

<a id="paper-3733"></a>

## Mosaic: Runtime-Efficient Multi-Agent Embodied Planning


### Images

![0d3f73303ee90f78a9eee36fc4216574413f1e575b1fd4da0b2d65755e652ae0.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/0d3f73303ee90f78a9eee36fc4216574413f1e575b1fd4da0b2d65755e652ae0.jpg)

![2801fec20d420f5b97cd83ac80de3255c352559a7396f9e31e3c658dc06d8747.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/2801fec20d420f5b97cd83ac80de3255c352559a7396f9e31e3c658dc06d8747.jpg)

![4721cc9523f966f5bec84e66648c2802bc5845ba9afe500107b198a64f86b8d8.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/4721cc9523f966f5bec84e66648c2802bc5845ba9afe500107b198a64f86b8d8.jpg)

![4aedb38d62b6d91c2348843224b9f38b05831694ff07fe2f3433e27eef4c5f09.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/4aedb38d62b6d91c2348843224b9f38b05831694ff07fe2f3433e27eef4c5f09.jpg)

![4e2d5a438207f1629a61d35926697e9eae7c8fedc3ae168cebcde25682060625.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/4e2d5a438207f1629a61d35926697e9eae7c8fedc3ae168cebcde25682060625.jpg)

![54c8de64d336e8bfdba4574326f474db24744c0afa62c06df022cc27e4a14630.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/54c8de64d336e8bfdba4574326f474db24744c0afa62c06df022cc27e4a14630.jpg)

![55a52366124ef40ab3812fd8a8bcd92710441a9b573f52c911eb5f47a9183c7c.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/55a52366124ef40ab3812fd8a8bcd92710441a9b573f52c911eb5f47a9183c7c.jpg)

![8f9b2a95014de0bfe912e05763211361f952bba2342d7a15d0feccc5280c90ba.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/8f9b2a95014de0bfe912e05763211361f952bba2342d7a15d0feccc5280c90ba.jpg)

![92fa5c9b3a8c991cacc7959f86686db7ce147a759e4af78a7d38b9d730cad01a.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/92fa5c9b3a8c991cacc7959f86686db7ce147a759e4af78a7d38b9d730cad01a.jpg)

![9c126a1afd7a293a29f8deb1c7a3ff8a617899a57497e3ab233390bb0594785e.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/9c126a1afd7a293a29f8deb1c7a3ff8a617899a57497e3ab233390bb0594785e.jpg)

![ac0b9c52cdaf853bfdb00967accbf22279bf6c588dd35b558dbb3afca0e7445e.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/ac0b9c52cdaf853bfdb00967accbf22279bf6c588dd35b558dbb3afca0e7445e.jpg)

![c7701589810c233705b8bb58798b8d7f9062fd95746e3ad1fcc2781cb0de76b9.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/c7701589810c233705b8bb58798b8d7f9062fd95746e3ad1fcc2781cb0de76b9.jpg)

![d6cd828a8fb7f3ba7fbf9462b07931317a4ae660c0a1b35fa08c3c9f43be658b.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/d6cd828a8fb7f3ba7fbf9462b07931317a4ae660c0a1b35fa08c3c9f43be658b.jpg)

![e06cb468ecbf9fb60c8f3cc9204aef231814818e4202ded002fef828ad3fb448.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/images/e06cb468ecbf9fb60c8f3cc9204aef231814818e4202ded002fef828ad3fb448.jpg)

### Tables

![029b7e90141ae2921c5208771e648807b1cda0a2e70b8710461015a1e6792e4c.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/029b7e90141ae2921c5208771e648807b1cda0a2e70b8710461015a1e6792e4c.jpg)

![1da029abc72b9a9bcf14504b401bc04a11c6b3edd8af5cfed13a48ddf7db2a21.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/1da029abc72b9a9bcf14504b401bc04a11c6b3edd8af5cfed13a48ddf7db2a21.jpg)

![40e59653ce8ddaf95bed310ab2a9ed934cd531a1b0727bd5996a19b190725b95.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/40e59653ce8ddaf95bed310ab2a9ed934cd531a1b0727bd5996a19b190725b95.jpg)

![41df5b8183a621c87a16bb28fee74acaf69a89f38a52abdb363f11fe48509505.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/41df5b8183a621c87a16bb28fee74acaf69a89f38a52abdb363f11fe48509505.jpg)

![593531f58fc7299c55567380e00e64ca4693e21237b1e759eb4f79f4b0ee4fbd.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/593531f58fc7299c55567380e00e64ca4693e21237b1e759eb4f79f4b0ee4fbd.jpg)

![635629fefb5c030294aea2cb3a4831c7a0a9ff38bc5ac2871138681660d61216.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/635629fefb5c030294aea2cb3a4831c7a0a9ff38bc5ac2871138681660d61216.jpg)

![6471b9a2cab1d1758193ac0911a3fa3625d6f11c122b03720a054213bf86bb0e.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/6471b9a2cab1d1758193ac0911a3fa3625d6f11c122b03720a054213bf86bb0e.jpg)

![7587534680915f7f5ad8db98eef6b7372581c20b0393d11e85d9ed5d2c209a00.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/7587534680915f7f5ad8db98eef6b7372581c20b0393d11e85d9ed5d2c209a00.jpg)

![88532fd58eabbd4b69b4ad1d5abb6b64c7286b7c838629cfab2c186778e30960.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/88532fd58eabbd4b69b4ad1d5abb6b64c7286b7c838629cfab2c186778e30960.jpg)

![93f346bae9ae4e548d311c7ade3914af6749ac76c94f316fdb1149f73bc3b9ba.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/93f346bae9ae4e548d311c7ade3914af6749ac76c94f316fdb1149f73bc3b9ba.jpg)

![9e9178de66ce7268783f2efedfb4db978e57fad9ceef37bb2dcf74998ba9c446.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/9e9178de66ce7268783f2efedfb4db978e57fad9ceef37bb2dcf74998ba9c446.jpg)

![b33c7f59e9ee6bf70ae0f7d84c6d1977c4424ff78bbf0381d106e98bee3d6b1e.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/b33c7f59e9ee6bf70ae0f7d84c6d1977c4424ff78bbf0381d106e98bee3d6b1e.jpg)

![b3e4d21980d0f29a79417693d9dc85593df6fabf123baaaf2f9ceb2b6df3a4a5.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/b3e4d21980d0f29a79417693d9dc85593df6fabf123baaaf2f9ceb2b6df3a4a5.jpg)

![b7eddc7ab395fb490ba8d70789280edb33f590bb8b88899697091a6b8d3f8589.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/b7eddc7ab395fb490ba8d70789280edb33f590bb8b88899697091a6b8d3f8589.jpg)

![c0e5aed85ee4d89b040f3f8d807a5c5bed26ed87e0e2ffd0484a92994692063e.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/c0e5aed85ee4d89b040f3f8d807a5c5bed26ed87e0e2ffd0484a92994692063e.jpg)

![cd998d971840167c0889f89354da90394cae16a45492d4d6582665ea8aa14b1c.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/cd998d971840167c0889f89354da90394cae16a45492d4d6582665ea8aa14b1c.jpg)

![d98e321aab1b2872e493b3c58bd0b424683eb6d411f85d8c1435d6088f98ae47.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/d98e321aab1b2872e493b3c58bd0b424683eb6d411f85d8c1435d6088f98ae47.jpg)

![e6a3c02ed92aa51c94f14799306e8c3c62e70aa81fb6d44c1007b86df53b8ea5.jpg](../icml_results/3773_Mosaic_%20Runtime-Efficient%20Multi-Agent%20Embodied%20Planning/tables/e6a3c02ed92aa51c94f14799306e8c3c62e70aa81fb6d44c1007b86df53b8ea5.jpg)

<a id="paper-3734"></a>

## Interpretability Driven Evolutionary Approach for the Design of Biological Sequences


### Images

![0d43036bc7125f6855cea2bb95ceadafe7033387f8418cf73bd4640c0eb38cbc.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/0d43036bc7125f6855cea2bb95ceadafe7033387f8418cf73bd4640c0eb38cbc.jpg)

![2e975f09340e5b88480158ee74c137e1d45da951dabfb81f648255c962d7246f.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/2e975f09340e5b88480158ee74c137e1d45da951dabfb81f648255c962d7246f.jpg)

![48f6d1dbb545dae5b197649dd04eaa49eb96b13ee3c7c5c5ff4d6c0d0acf9062.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/48f6d1dbb545dae5b197649dd04eaa49eb96b13ee3c7c5c5ff4d6c0d0acf9062.jpg)

![4d71cb21b6e190cb723051ded1b0f8f4b5b51797ff9f552107ff2e0db5670a09.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/4d71cb21b6e190cb723051ded1b0f8f4b5b51797ff9f552107ff2e0db5670a09.jpg)

![559950aed33cacdf224e3c0cd549302e51799c88ba0bfecdd02b2c0006f094f9.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/559950aed33cacdf224e3c0cd549302e51799c88ba0bfecdd02b2c0006f094f9.jpg)

![6045a6438c63ac1679ea0989e9845b09d5f141ed7fc3e5991bf3397ca20fa561.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/6045a6438c63ac1679ea0989e9845b09d5f141ed7fc3e5991bf3397ca20fa561.jpg)

![8cdd46ae1dba7391d9551f5ba6c7323d2b69fd824591d596541911c47c4df91a.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/8cdd46ae1dba7391d9551f5ba6c7323d2b69fd824591d596541911c47c4df91a.jpg)

![98471e1d3d9e22335fdeefa5ad3bd9eccdf2b467b5d97fc327b201a4e3cab8af.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/98471e1d3d9e22335fdeefa5ad3bd9eccdf2b467b5d97fc327b201a4e3cab8af.jpg)

![b9255865687f6581437f7f6bea18f3223fcddf9eaf8088d4aeb152a5c8bc9d16.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/b9255865687f6581437f7f6bea18f3223fcddf9eaf8088d4aeb152a5c8bc9d16.jpg)

![bcb542a0a721a093a7e367e1b779cea9173d6c0b61d0f6f101192decc09a8432.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/bcb542a0a721a093a7e367e1b779cea9173d6c0b61d0f6f101192decc09a8432.jpg)

![d138d6c5f32d9aabd3e742188f9582d1578ecff3a5c8ba9426687991fc706a2b.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/d138d6c5f32d9aabd3e742188f9582d1578ecff3a5c8ba9426687991fc706a2b.jpg)

![e1bd96826503179432353c63c35bfa44878dc0b14f325bf12e56759897e17fb0.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/e1bd96826503179432353c63c35bfa44878dc0b14f325bf12e56759897e17fb0.jpg)

![e1f99921c20ae3102f32ccf767047a1006bca9889df515be74485c96b0cac9a1.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/e1f99921c20ae3102f32ccf767047a1006bca9889df515be74485c96b0cac9a1.jpg)

![e4053465b581875ac41dce314d74f6b2e18166dc9b8697ae97b80962fbbfdf9d.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/e4053465b581875ac41dce314d74f6b2e18166dc9b8697ae97b80962fbbfdf9d.jpg)

![e81a7dd518a7c28d7d64c032cd13b02cfc2e0d1e46461e6290d5678dcb851a6b.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/images/e81a7dd518a7c28d7d64c032cd13b02cfc2e0d1e46461e6290d5678dcb851a6b.jpg)

### Tables

![1df5c76def92c107593e887c26e74258181ba6673fae5b8e92c2e0c4ea90e790.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/1df5c76def92c107593e887c26e74258181ba6673fae5b8e92c2e0c4ea90e790.jpg)

![1f189846aa217c13e8ca6c5e9bf06831ce63b5f98ec7ab1869e218cc0ba8b6e0.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/1f189846aa217c13e8ca6c5e9bf06831ce63b5f98ec7ab1869e218cc0ba8b6e0.jpg)

![3a1dd8ea465f56583ae6495abbb1db15e2a6ac7ca6b574de4b08a5906e3cbf61.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/3a1dd8ea465f56583ae6495abbb1db15e2a6ac7ca6b574de4b08a5906e3cbf61.jpg)

![76f3ffc52fd1e3c92d61dd8b16ada3fcff3afb9ae9af683eb27a4911f05c9374.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/76f3ffc52fd1e3c92d61dd8b16ada3fcff3afb9ae9af683eb27a4911f05c9374.jpg)

![778f7342735ec367f50ab5d33ba7003977f9b70a79cba6cf63d7b945586c0a57.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/778f7342735ec367f50ab5d33ba7003977f9b70a79cba6cf63d7b945586c0a57.jpg)

![9396af855888fd9f3fb7aefee0627c0d27aea0f066a4484ae1eda357e22b69d5.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/9396af855888fd9f3fb7aefee0627c0d27aea0f066a4484ae1eda357e22b69d5.jpg)

![9c003c8f9ee61cc0ca0296380f995f8fb2af999214bf687fcd5bb4347ac6ea63.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/9c003c8f9ee61cc0ca0296380f995f8fb2af999214bf687fcd5bb4347ac6ea63.jpg)

![d3db2aff6533365df249e70c6c4db5c50671d67fe23cdec9545a272ab3642739.jpg](../icml_results/3774_Interpretability%20Driven%20Evolutionary%20Approach%20for%20the%20Design%20of%20Biological%20Sequences/tables/d3db2aff6533365df249e70c6c4db5c50671d67fe23cdec9545a272ab3642739.jpg)

<a id="paper-3735"></a>

## Continual Segmentation under Joint Nonstationarity


### Images

![18a54939a5309e74d1d45936db1a10ac2145cb60e876032ff2ec03592b133f2a.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/images/18a54939a5309e74d1d45936db1a10ac2145cb60e876032ff2ec03592b133f2a.jpg)

![a573446c6e4abd6dfb0c43a81f9342b54f899f784f130060ab6b96e7d0c8bb26.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/images/a573446c6e4abd6dfb0c43a81f9342b54f899f784f130060ab6b96e7d0c8bb26.jpg)

![a706cbbb2b9bbc544df67ea0900871ae09437184aff95677b9162f8525ec084c.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/images/a706cbbb2b9bbc544df67ea0900871ae09437184aff95677b9162f8525ec084c.jpg)

![e4806548dfe8ba6a91db9709c48532d6ecbf1b9d1aac1dc015c5af9b4faa004b.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/images/e4806548dfe8ba6a91db9709c48532d6ecbf1b9d1aac1dc015c5af9b4faa004b.jpg)

![e7a2c1f52ee7b03134d987cc5821b55bade9cf8976033ca9d796074e5c2b1854.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/images/e7a2c1f52ee7b03134d987cc5821b55bade9cf8976033ca9d796074e5c2b1854.jpg)

![f7cad79c3778162c59964ac1b7663b1a2a1f9a5edbe90a8cd6c08c5a5ff020f6.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/images/f7cad79c3778162c59964ac1b7663b1a2a1f9a5edbe90a8cd6c08c5a5ff020f6.jpg)

### Tables

![0612b7ff8b5153eff1b3abc8fb0825f9f164c9d248756db1a9fe1adace22361f.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/0612b7ff8b5153eff1b3abc8fb0825f9f164c9d248756db1a9fe1adace22361f.jpg)

![085e69ede52001fec542fd9c5d6a4527e18381fc920b4f450f621df4b9aad058.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/085e69ede52001fec542fd9c5d6a4527e18381fc920b4f450f621df4b9aad058.jpg)

![08b8188e2b3bbbf5cce68458f7db257f3ef2b196560b30a7564b39c0263ebc83.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/08b8188e2b3bbbf5cce68458f7db257f3ef2b196560b30a7564b39c0263ebc83.jpg)

![1013124a01d6be39784e76b3a73e6d4af1f314d788679daa4e67ae8a08d06607.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/1013124a01d6be39784e76b3a73e6d4af1f314d788679daa4e67ae8a08d06607.jpg)

![143962a3f52917a173dc512c1e2f0ae4a95a3081ba3f8e300d1d3e90ad195740.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/143962a3f52917a173dc512c1e2f0ae4a95a3081ba3f8e300d1d3e90ad195740.jpg)

![22b05ba1afe64d820cdec93db8cf9c00beb27c5ebdc21ee354146c3ddc5a9057.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/22b05ba1afe64d820cdec93db8cf9c00beb27c5ebdc21ee354146c3ddc5a9057.jpg)

![242d21e62248d281dc68b7027cb472f85d6de4a294a6c45d6153bfa03858cf04.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/242d21e62248d281dc68b7027cb472f85d6de4a294a6c45d6153bfa03858cf04.jpg)

![2d09c7fae430a27c3e76384d394d13ae57a7fe99bda8b6b36b046e41e850df03.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/2d09c7fae430a27c3e76384d394d13ae57a7fe99bda8b6b36b046e41e850df03.jpg)

![2eaff90ec58c0c1db2efbe8cf4d6b7ac58870a21a1229928cffb98975ce29918.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/2eaff90ec58c0c1db2efbe8cf4d6b7ac58870a21a1229928cffb98975ce29918.jpg)

![2f8719d9a05b77788f0ab9ee6eeadb6e6bf2ce2dfc69934c12892c879d5a3ee2.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/2f8719d9a05b77788f0ab9ee6eeadb6e6bf2ce2dfc69934c12892c879d5a3ee2.jpg)

![3e276893d2ea4f8bfb5f30d4432c2e50f496977b9a36a7af7a3b56fb7fc437dc.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/3e276893d2ea4f8bfb5f30d4432c2e50f496977b9a36a7af7a3b56fb7fc437dc.jpg)

![3f7149d9eee37f78480fadea4139d60de33626a019ecd4434eec927274d8de31.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/3f7149d9eee37f78480fadea4139d60de33626a019ecd4434eec927274d8de31.jpg)

![4469229c266d03e570bc29c87fae69d786defe45a1a7343351a2c9f13624d61b.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/4469229c266d03e570bc29c87fae69d786defe45a1a7343351a2c9f13624d61b.jpg)

![46dd183282f7e40a1784035507c8f52edb42dcca82802e16951dcbcfe80ff50d.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/46dd183282f7e40a1784035507c8f52edb42dcca82802e16951dcbcfe80ff50d.jpg)

![4bed87fbdfdde8ca0827c6d8343d6a6159e9aef13074ef5238b3725bfcb97128.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/4bed87fbdfdde8ca0827c6d8343d6a6159e9aef13074ef5238b3725bfcb97128.jpg)

![52dbd8b71a6415d0d1aad44949d15ca5e6d74d2729e947eefa1aa9eb9b7984c1.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/52dbd8b71a6415d0d1aad44949d15ca5e6d74d2729e947eefa1aa9eb9b7984c1.jpg)

![56d065a35359337805b6eb2d18f0a7965c73103fd327b6d6de6d99ec53051d37.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/56d065a35359337805b6eb2d18f0a7965c73103fd327b6d6de6d99ec53051d37.jpg)

![5cb8af52e44099cc673055c1b0fded56fd8c23cf4299ee48691c03f2380cc415.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/5cb8af52e44099cc673055c1b0fded56fd8c23cf4299ee48691c03f2380cc415.jpg)

![5db4f60d01cf99dfb3ff89bb0ebe47b5d1a3d39228796f40f9eac65b3cead063.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/5db4f60d01cf99dfb3ff89bb0ebe47b5d1a3d39228796f40f9eac65b3cead063.jpg)

![6126dfc38a2c6b0530e08d49f012af654a56047056e0172384272b1282e72726.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/6126dfc38a2c6b0530e08d49f012af654a56047056e0172384272b1282e72726.jpg)

![63504323b7a2fa0cd654c6b82721ddfdb349e38530a83f2384f7cc44a339cd1d.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/63504323b7a2fa0cd654c6b82721ddfdb349e38530a83f2384f7cc44a339cd1d.jpg)

![6979318f2442da6ce112d9ecabf065f33a2cf82d8f27b1b06fe29137c6d81526.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/6979318f2442da6ce112d9ecabf065f33a2cf82d8f27b1b06fe29137c6d81526.jpg)

![69c52190ffada93fda5d01929855c5a95af3953dfadf6187353fcaa869c15ebf.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/69c52190ffada93fda5d01929855c5a95af3953dfadf6187353fcaa869c15ebf.jpg)

![6e4df5a3888a77990a043abf65c1f23557a6a2ddd2e506fe30a7f81079b4cec3.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/6e4df5a3888a77990a043abf65c1f23557a6a2ddd2e506fe30a7f81079b4cec3.jpg)

![7f2afd7a6b6c2d633bc0c3da30adb9da8195f0b4ba6fe34438d6177f532f5271.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/7f2afd7a6b6c2d633bc0c3da30adb9da8195f0b4ba6fe34438d6177f532f5271.jpg)

![828760029485c810d65e79a870fd01ed9a01fa7d73980131802f61512b84ddd1.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/828760029485c810d65e79a870fd01ed9a01fa7d73980131802f61512b84ddd1.jpg)

![8805f1919e572c9422cc47b15b3e171fe777a3981accdfb7528a42f32a9e6610.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/8805f1919e572c9422cc47b15b3e171fe777a3981accdfb7528a42f32a9e6610.jpg)

![8a80c8ab8a3c0f6f0b75d54360ce861f32392743ccf9b12b48361f83aad149ad.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/8a80c8ab8a3c0f6f0b75d54360ce861f32392743ccf9b12b48361f83aad149ad.jpg)

![961bd6c23ae328c549d5c585f6a3a31b0256a214c3456a27d33acdaffb04217f.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/961bd6c23ae328c549d5c585f6a3a31b0256a214c3456a27d33acdaffb04217f.jpg)

![9695dc3b068022b75d1fc52287aa97602a498188957c740701c53a175aa9b73f.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/9695dc3b068022b75d1fc52287aa97602a498188957c740701c53a175aa9b73f.jpg)

![a78e3919a7469b0a71ffec3313650770c0466560bb20b1b0ab3a59daf6cae37b.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/a78e3919a7469b0a71ffec3313650770c0466560bb20b1b0ab3a59daf6cae37b.jpg)

![ae5de61f051645bba6e7dce9175f90e2c10be6ccf5d86e9656c0065e4ebe76a8.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/ae5de61f051645bba6e7dce9175f90e2c10be6ccf5d86e9656c0065e4ebe76a8.jpg)

![b2623f655962f13e5f2ea24bc16d32eb4d9aa7638d3aad5b574b340f158faa79.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/b2623f655962f13e5f2ea24bc16d32eb4d9aa7638d3aad5b574b340f158faa79.jpg)

![b2e2afc641a4af49006b82c2bdc6c1e1856b38b8c588be6106499921bc112c61.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/b2e2afc641a4af49006b82c2bdc6c1e1856b38b8c588be6106499921bc112c61.jpg)

![b4d6cc309a2e367ff689aabcf74118020e58f79abd73fe7230d0426eed1c49d2.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/b4d6cc309a2e367ff689aabcf74118020e58f79abd73fe7230d0426eed1c49d2.jpg)

![baa352a9533f025122160fe5c92418233648ff8313ca0a2f4b47c370e47380f4.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/baa352a9533f025122160fe5c92418233648ff8313ca0a2f4b47c370e47380f4.jpg)

![bd51684bd0d38069868d0ee1dabfff650f56d025b643280509bc014c55331515.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/bd51684bd0d38069868d0ee1dabfff650f56d025b643280509bc014c55331515.jpg)

![c4e8ad0610f52b9c0d3dc574104310075a807a4756c3b963eb93ff171d970f69.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/c4e8ad0610f52b9c0d3dc574104310075a807a4756c3b963eb93ff171d970f69.jpg)

![d3263799e98b03f0d086fc82b31a56563c48236496562064e9507b778199b8dd.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/d3263799e98b03f0d086fc82b31a56563c48236496562064e9507b778199b8dd.jpg)

![d3def54a7fd5ca296aeab398c959f1eb5b0987c5a302abcc719f74ce451afd71.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/d3def54a7fd5ca296aeab398c959f1eb5b0987c5a302abcc719f74ce451afd71.jpg)

![e34cc5dcb4bdf229fe2ea2a66fbdc4f31f4af62645469d3184fbd4f946f2f5d8.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/e34cc5dcb4bdf229fe2ea2a66fbdc4f31f4af62645469d3184fbd4f946f2f5d8.jpg)

![e7f7ca739557e8548ade43360f959b322765dccaa35ca7814556b695b15a9d6f.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/e7f7ca739557e8548ade43360f959b322765dccaa35ca7814556b695b15a9d6f.jpg)

![eea316523e98ce6d54e89ed0c766b60cc25b7f69f2b35984a56e418b05ba75a5.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/eea316523e98ce6d54e89ed0c766b60cc25b7f69f2b35984a56e418b05ba75a5.jpg)

![f4074075173bf4119254373454d7565b94eed30dc4168db2616fdb9fa82c7c09.jpg](../icml_results/3775_Continual%20Segmentation%20under%20Joint%20Nonstationarity/tables/f4074075173bf4119254373454d7565b94eed30dc4168db2616fdb9fa82c7c09.jpg)

<a id="paper-3736"></a>

## Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation


### Images

![09f05addf4c38d0fbcaf6deef0e16bf3eb851e1bf453dee0214cbb335f342818.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/09f05addf4c38d0fbcaf6deef0e16bf3eb851e1bf453dee0214cbb335f342818.jpg)

![139be597e990a72ca921fb926b5943513ee0df5833b4a45d646fbe6b55e29428.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/139be597e990a72ca921fb926b5943513ee0df5833b4a45d646fbe6b55e29428.jpg)

![1b48878f7bae151c32a585abd57fa6084987b879a6777bda29c7dcb9f881df9f.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/1b48878f7bae151c32a585abd57fa6084987b879a6777bda29c7dcb9f881df9f.jpg)

![21e54e56611fb0429a5a9d6c87568d976f8a04677444d48e6887acb4193d5661.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/21e54e56611fb0429a5a9d6c87568d976f8a04677444d48e6887acb4193d5661.jpg)

![85cf7319f9a11e6580c3361232f696e3ba76cc1933d3ab0c443539f297f1c9e1.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/85cf7319f9a11e6580c3361232f696e3ba76cc1933d3ab0c443539f297f1c9e1.jpg)

![a475e0d70c5197a5275b09c22fedbec6f00515d1febc420caf2522acf5d8102e.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/a475e0d70c5197a5275b09c22fedbec6f00515d1febc420caf2522acf5d8102e.jpg)

![a8060c1677b366a131b00841ae262fc99fc6fa9a4e1fd3a8e0e836dcd7af27e5.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/a8060c1677b366a131b00841ae262fc99fc6fa9a4e1fd3a8e0e836dcd7af27e5.jpg)

![d2bd67fb563094bb3a6736083a75055c93b0e05e3aa7f80032d622289190227d.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/d2bd67fb563094bb3a6736083a75055c93b0e05e3aa7f80032d622289190227d.jpg)

![e93d8fb8e4cb3ba2bafe039859ba24ca28da04be8c5a22eec9ea0dd613045d52.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/e93d8fb8e4cb3ba2bafe039859ba24ca28da04be8c5a22eec9ea0dd613045d52.jpg)

![f05bc9064631ab750141a85c2abdf4d9d6900a6508136379dfa36d18a7418956.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/images/f05bc9064631ab750141a85c2abdf4d9d6900a6508136379dfa36d18a7418956.jpg)

### Tables

![5401ad5dd8f8e97dce5b07e15197531d9f9fd17f83855a4170aea40fb55031a9.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/5401ad5dd8f8e97dce5b07e15197531d9f9fd17f83855a4170aea40fb55031a9.jpg)

![62b26bf37af331599f426924a3efa56600c78b23777ff74b63d87c3d0ba4ee89.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/62b26bf37af331599f426924a3efa56600c78b23777ff74b63d87c3d0ba4ee89.jpg)

![64ab640e35884fdf066e0d5c3b8feccd9b7c0bab70f67ca11a48edd98072d931.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/64ab640e35884fdf066e0d5c3b8feccd9b7c0bab70f67ca11a48edd98072d931.jpg)

![8138d37022bc1da04289ac446552713a6f3e7f680dce2bcbe22a833a5fc03b21.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/8138d37022bc1da04289ac446552713a6f3e7f680dce2bcbe22a833a5fc03b21.jpg)

![8b9570b9752fe2b911044eccb3919a84d712006e139edb326259273d50f701a7.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/8b9570b9752fe2b911044eccb3919a84d712006e139edb326259273d50f701a7.jpg)

![92adfda9efa634eb257fbf1c08117c0708abbca0e3860fe6d96bd22ed5b632d0.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/92adfda9efa634eb257fbf1c08117c0708abbca0e3860fe6d96bd22ed5b632d0.jpg)

![a61352b72abdc2b301f86387043fbebf73b37c49ab4d457fd1675fd52def9654.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/a61352b72abdc2b301f86387043fbebf73b37c49ab4d457fd1675fd52def9654.jpg)

![c83612d7624cf3e6e258f0be74a96ccdf13ff78855afb1b57e724536b8c03a52.jpg](../icml_results/3776_Quartet%20II_%20Accurate%20LLM%20Pre-Training%20in%20NVFP4%20by%20Improved%20Unbiased%20Gradient%20Estimation/tables/c83612d7624cf3e6e258f0be74a96ccdf13ff78855afb1b57e724536b8c03a52.jpg)

<a id="paper-3737"></a>

## Decouple and Cache: KV Cache Construction for Streaming Video Understanding


### Images

![062e9701f4b48bb7468ef0f75be1490af496685ed9c5c84a2048de6a2bf4f88f.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/062e9701f4b48bb7468ef0f75be1490af496685ed9c5c84a2048de6a2bf4f88f.jpg)

![0d57d1ee91da98e278fdc4faa4feb66ac460bc73ab2c9ca6bb406a105bf01c7b.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/0d57d1ee91da98e278fdc4faa4feb66ac460bc73ab2c9ca6bb406a105bf01c7b.jpg)

![1294ab4026484eadb8c33ae43c91bd23d4dee4bde3791c06290958ee3ac51597.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/1294ab4026484eadb8c33ae43c91bd23d4dee4bde3791c06290958ee3ac51597.jpg)

![191c1680b1e53476905323b1de1f2b977bc68af014efab0ed412588b22d77a51.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/191c1680b1e53476905323b1de1f2b977bc68af014efab0ed412588b22d77a51.jpg)

![36e810e03370799acf7cc5a32d4fe1d30f303da8e9504993d94f39b24c36305e.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/36e810e03370799acf7cc5a32d4fe1d30f303da8e9504993d94f39b24c36305e.jpg)

![81ea5871d94fdf2a8a406e4009f222d78ae7b5e0a1361588aec5758f52ccbdef.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/81ea5871d94fdf2a8a406e4009f222d78ae7b5e0a1361588aec5758f52ccbdef.jpg)

![94c1bdf0d1279e364b95741f0efe18ad116144d8b1480773b4d9390d8e07953c.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/94c1bdf0d1279e364b95741f0efe18ad116144d8b1480773b4d9390d8e07953c.jpg)

![bb91e76574285997843d32f0fd1c76d9ac70d77dd45e857d895f38fd707abc0d.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/bb91e76574285997843d32f0fd1c76d9ac70d77dd45e857d895f38fd707abc0d.jpg)

![c7390b6282e7c8b4d33c31fb7a3d609a4950490e17184eded96093ac81e71b06.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/c7390b6282e7c8b4d33c31fb7a3d609a4950490e17184eded96093ac81e71b06.jpg)

![e7f8768b7390be77b29073ee2e068a8f4e1933e918af3d0a634046621d944eff.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/images/e7f8768b7390be77b29073ee2e068a8f4e1933e918af3d0a634046621d944eff.jpg)

### Tables

![01cb72382abdaf179234cce295e205d15e3d641111200319c10b46c11984ad25.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/01cb72382abdaf179234cce295e205d15e3d641111200319c10b46c11984ad25.jpg)

![1403c729ae30aea5645088082bee9de0ce438622483d81f28ad8b62a23c61da5.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/1403c729ae30aea5645088082bee9de0ce438622483d81f28ad8b62a23c61da5.jpg)

![1445da26c55e579204143420707541a9950da02e8c13c34c5a41f0a77e0ad549.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/1445da26c55e579204143420707541a9950da02e8c13c34c5a41f0a77e0ad549.jpg)

![179d998c9c155d232bf52f052901fc175c4195636eec713c8e287fa6eb28054d.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/179d998c9c155d232bf52f052901fc175c4195636eec713c8e287fa6eb28054d.jpg)

![224c8856df0081965a30a5da6656baa08976e4812443beab5fa4572734e73a3f.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/224c8856df0081965a30a5da6656baa08976e4812443beab5fa4572734e73a3f.jpg)

![66ffaf77142c63090b5451993ce80c368941eee5360d668683afda8cee6fcbe0.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/66ffaf77142c63090b5451993ce80c368941eee5360d668683afda8cee6fcbe0.jpg)

![6f836ee05141d910110beb713480ea23a596fcd71cc2cd4584ab69b86c877bc7.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/6f836ee05141d910110beb713480ea23a596fcd71cc2cd4584ab69b86c877bc7.jpg)

![87566fc7f7eb80a41e72f8794da98d2707f2f2a1b588a9606adf3d20841721c2.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/87566fc7f7eb80a41e72f8794da98d2707f2f2a1b588a9606adf3d20841721c2.jpg)

![88fef998ddac288a7b30d8d7ec76c89901a6725546765e88799b2d78b23d6f04.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/88fef998ddac288a7b30d8d7ec76c89901a6725546765e88799b2d78b23d6f04.jpg)

![afe07cd4f43eec9ddc406c301928a4d748a77b9e382121ca8e505806d6bf7c50.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/afe07cd4f43eec9ddc406c301928a4d748a77b9e382121ca8e505806d6bf7c50.jpg)

![f05e42f8a6107ce4f5e9360c04ce1845f209ee0b50e9f0e68af10364bd2dc163.jpg](../icml_results/3777_Decouple%20and%20Cache_%20KV%20Cache%20Construction%20for%20Streaming%20Video%20Understanding/tables/f05e42f8a6107ce4f5e9360c04ce1845f209ee0b50e9f0e68af10364bd2dc163.jpg)

<a id="paper-3738"></a>

## Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox


### Images

![2d1a5cb60428962175d5f4274d6a15a0234dba11b9e5b412d23ce46c3d46b1b3.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/2d1a5cb60428962175d5f4274d6a15a0234dba11b9e5b412d23ce46c3d46b1b3.jpg)

![36cc137c962b145a1fc3deef97859380984f61d3df7d88b323180a490b710a2a.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/36cc137c962b145a1fc3deef97859380984f61d3df7d88b323180a490b710a2a.jpg)

![40b9563519d5cf59c0cc547b82fde8929dd176d0409d0a68fb30afd29d7f1c6d.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/40b9563519d5cf59c0cc547b82fde8929dd176d0409d0a68fb30afd29d7f1c6d.jpg)

![59a6ada8c1dcb15a8f6594b33f16372698ee33b2ceaca6c84ac9bbe3d920ab36.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/59a6ada8c1dcb15a8f6594b33f16372698ee33b2ceaca6c84ac9bbe3d920ab36.jpg)

![62a5fb6a37cf76c80bbaf1ca26a7473516f8e2da8971e35153ec9bf3708a46eb.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/62a5fb6a37cf76c80bbaf1ca26a7473516f8e2da8971e35153ec9bf3708a46eb.jpg)

![80595f936931471f154d57869f83b40f127fbd3f2e139c4a440889d2cfa1990d.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/80595f936931471f154d57869f83b40f127fbd3f2e139c4a440889d2cfa1990d.jpg)

![8b1e99dfc52eb0cf4a647405d7d4d2f8976618a0369813853e97bc04d60277c1.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/8b1e99dfc52eb0cf4a647405d7d4d2f8976618a0369813853e97bc04d60277c1.jpg)

![a4e05d4a041274ac5f1be0f96fcae543808c1fd740dd3838e32f36551b6543bf.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/a4e05d4a041274ac5f1be0f96fcae543808c1fd740dd3838e32f36551b6543bf.jpg)

![a90b6f693f8073f23e980f4b75298d0d9b0ba8218f3bd91ab917b3c50253374e.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/images/a90b6f693f8073f23e980f4b75298d0d9b0ba8218f3bd91ab917b3c50253374e.jpg)

### Tables

![05bde44b6c2979b505dccd02d928d7c250bf8b48f0be43569e36775f9fa8b8c3.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/05bde44b6c2979b505dccd02d928d7c250bf8b48f0be43569e36775f9fa8b8c3.jpg)

![15e861a75c33fed935f42e592447310b6cb1923292a407b682fceb47014e66c5.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/15e861a75c33fed935f42e592447310b6cb1923292a407b682fceb47014e66c5.jpg)

![3ae9e668ff0fd281c7948011ba1a91326049ff30a687bb84fb9894b150e869fa.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/3ae9e668ff0fd281c7948011ba1a91326049ff30a687bb84fb9894b150e869fa.jpg)

![4fadca293b15e224c957bf6ebddd0d9bb209d2ca5a28c60ac793e7f290b545c3.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/4fadca293b15e224c957bf6ebddd0d9bb209d2ca5a28c60ac793e7f290b545c3.jpg)

![680bc92a86aeb1a9bcb2d61c806acd8a8b03aece6b1f64acbd20ccc2538ac403.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/680bc92a86aeb1a9bcb2d61c806acd8a8b03aece6b1f64acbd20ccc2538ac403.jpg)

![68f495a162fe06118d414ce605c6df11cffe8d3ccb24b35d210cafc9caf6becd.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/68f495a162fe06118d414ce605c6df11cffe8d3ccb24b35d210cafc9caf6becd.jpg)

![cefc92de61e8dcf6bc230a6b5b5712d6e0ec156525a66c2a6f89434307d3f5bd.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/cefc92de61e8dcf6bc230a6b5b5712d6e0ec156525a66c2a6f89434307d3f5bd.jpg)

![d0e4af8be66adb2c6cdb05d41ba578701bd14970fd5b7d3649c1cc3d74812085.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/d0e4af8be66adb2c6cdb05d41ba578701bd14970fd5b7d3649c1cc3d74812085.jpg)

![d79ea2d9740af82f43bb8fb8fe24c47705390a79fe1f86ad5bdb8883fc2f175d.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/d79ea2d9740af82f43bb8fb8fe24c47705390a79fe1f86ad5bdb8883fc2f175d.jpg)

![d805de4578a8f77b878c82c8686b69e1aafca2febdcec187eb8f25941e3a4434.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/d805de4578a8f77b878c82c8686b69e1aafca2febdcec187eb8f25941e3a4434.jpg)

![f295026b4da7088728272783b1d8c335870c1ef5a57a7b26f14823f3f6c4844d.jpg](../icml_results/3778_Do%20Audio%20LLMs%20Listen%20or%20Read_%20Analyzing%20and%20Mitigating%20Paralinguistic%20Failures%20with%20VoxParadox/tables/f295026b4da7088728272783b1d8c335870c1ef5a57a7b26f14823f3f6c4844d.jpg)

<a id="paper-3739"></a>

## Unveiling the Visual Counting Bottleneck in Vision-Language Models


### Images

![013313b52f80652fe7721f755857d59da7b1fcb57d49fe259e164be4f0b03513.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/013313b52f80652fe7721f755857d59da7b1fcb57d49fe259e164be4f0b03513.jpg)

![337aa070295b28f7d5bce19e65541a53357ae8fe0d74691f2e34302ec2e98d29.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/337aa070295b28f7d5bce19e65541a53357ae8fe0d74691f2e34302ec2e98d29.jpg)

![41fba1e75be4234463c67ff236ab39222d6e2ed746a0be5b12a03891a4f39dff.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/41fba1e75be4234463c67ff236ab39222d6e2ed746a0be5b12a03891a4f39dff.jpg)

![66cfd5ddf3119c286b98e55575bc4a380b0f4f9b0600cbffaf962df33079cb44.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/66cfd5ddf3119c286b98e55575bc4a380b0f4f9b0600cbffaf962df33079cb44.jpg)

![8dc092d218875ed156be7624405c37b1835265fe8f0c02c689d3aae0a386dcd5.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/8dc092d218875ed156be7624405c37b1835265fe8f0c02c689d3aae0a386dcd5.jpg)

![a6a29e71d53d6d5c2006dbd231b8716df94ef8195e04fc3f0d7b90c95e19dc11.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/a6a29e71d53d6d5c2006dbd231b8716df94ef8195e04fc3f0d7b90c95e19dc11.jpg)

![aac55359cdf6039a5d7843900458bd301ffa2eac4898675ad9ec3a732e0daf80.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/aac55359cdf6039a5d7843900458bd301ffa2eac4898675ad9ec3a732e0daf80.jpg)

![af1fcca5d110a705edb39678ab23d0f528f6f12d70d6529bcf74681c613b321d.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/af1fcca5d110a705edb39678ab23d0f528f6f12d70d6529bcf74681c613b321d.jpg)

![b81d6763084b2bfa56f4ec8c57acdc5217b7bad97f74f302ebb5bd8e810ba800.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/b81d6763084b2bfa56f4ec8c57acdc5217b7bad97f74f302ebb5bd8e810ba800.jpg)

![c8f8e746df64e409f939e916db278c031800d1ee3c2c2c4099d06d77fdac50eb.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/c8f8e746df64e409f939e916db278c031800d1ee3c2c2c4099d06d77fdac50eb.jpg)

![cc4d04f77e2832b6a5ab55ea7c41b5a7ef381c55b83bf8e5e9d929c05300f427.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/cc4d04f77e2832b6a5ab55ea7c41b5a7ef381c55b83bf8e5e9d929c05300f427.jpg)

![d60af25e56b3dd67b959677f94a7b966baf46f0b04586ec9549a337815d9246f.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/d60af25e56b3dd67b959677f94a7b966baf46f0b04586ec9549a337815d9246f.jpg)

![e8134d72bacf620694acc369a02165204b95f7ca322037c7df4981a4b6f1d465.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/e8134d72bacf620694acc369a02165204b95f7ca322037c7df4981a4b6f1d465.jpg)

![f02e0f819023b73b31cc38c8c385b190312892826308aa46d62755df4579cbdd.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/f02e0f819023b73b31cc38c8c385b190312892826308aa46d62755df4579cbdd.jpg)

![fbe85ffb18881f69e457ced36a25b4df0fd9beda837818f396ea9d07b7ad6f67.jpg](../icml_results/3779_Unveiling%20the%20Visual%20Counting%20Bottleneck%20in%20Vision-Language%20Models/images/fbe85ffb18881f69e457ced36a25b4df0fd9beda837818f396ea9d07b7ad6f67.jpg)

<a id="paper-3740"></a>

## Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks


### Images

![11c516703ce06248c2fe89495222aa917f9ef3b732b3050af74186f124e4b65d.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/11c516703ce06248c2fe89495222aa917f9ef3b732b3050af74186f124e4b65d.jpg)

![15fdf99a95bd99407db3d3fe2d835da0cb8ccb0cbdf7e6968b2c4e1afa0d3f67.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/15fdf99a95bd99407db3d3fe2d835da0cb8ccb0cbdf7e6968b2c4e1afa0d3f67.jpg)

![1e75ee519d0a884fe453788049f61013be6c2f3995feee5db30451ed675d9e1f.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/1e75ee519d0a884fe453788049f61013be6c2f3995feee5db30451ed675d9e1f.jpg)

![24a1e7e9057fb44681d2eb44e5e1a23b967209ba7136d68e4dff782fcbcf85b7.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/24a1e7e9057fb44681d2eb44e5e1a23b967209ba7136d68e4dff782fcbcf85b7.jpg)

![24a924dc889e4623e0c09b4dd363b652174d18df3a8a6f2f581416face88a900.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/24a924dc889e4623e0c09b4dd363b652174d18df3a8a6f2f581416face88a900.jpg)

![4b00c9674fab92e3c21ba9dcd1981eaf8108648e83421ce5073b124a936fde20.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/4b00c9674fab92e3c21ba9dcd1981eaf8108648e83421ce5073b124a936fde20.jpg)

![81559ddbcd96261449e334417be8c56663e949547cd3913819ac11e94e3149f7.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/81559ddbcd96261449e334417be8c56663e949547cd3913819ac11e94e3149f7.jpg)

![919f1bfa3deff9fcaad581082520ca59428a468e3b951eda3558d007a0aebbdb.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/919f1bfa3deff9fcaad581082520ca59428a468e3b951eda3558d007a0aebbdb.jpg)

![a90358ba2d893ec286b892a6aff5cd0b504cded9a979580399614ee5d3d70cc2.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/a90358ba2d893ec286b892a6aff5cd0b504cded9a979580399614ee5d3d70cc2.jpg)

![aaef8e0786be23e66b3e7de3d4a768755820bea8df4616526f6a62c0fd5bc5fc.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/aaef8e0786be23e66b3e7de3d4a768755820bea8df4616526f6a62c0fd5bc5fc.jpg)

![b92c27ea7c1412bd7c8f5ae4685bc3b94acb9b07b95e570d474dd80776781897.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/b92c27ea7c1412bd7c8f5ae4685bc3b94acb9b07b95e570d474dd80776781897.jpg)

![bef05b5db5912387ac9d5f8aca053278de78d5ada958eb2e447e8dc41830570b.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/bef05b5db5912387ac9d5f8aca053278de78d5ada958eb2e447e8dc41830570b.jpg)

![bf7c9457a6d89996321a0f0c9fb587a748d4398a095852b50c646be34359c205.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/bf7c9457a6d89996321a0f0c9fb587a748d4398a095852b50c646be34359c205.jpg)

![c3357448c4a212790b3d20a8dc04e860a8e07b347c9c9a545756b80e4d7f373f.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/c3357448c4a212790b3d20a8dc04e860a8e07b347c9c9a545756b80e4d7f373f.jpg)

![d4d02f6695dc73239350746e40e80cf0b3d6aa951f8e281a11ee6e8524aa9f0f.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/d4d02f6695dc73239350746e40e80cf0b3d6aa951f8e281a11ee6e8524aa9f0f.jpg)

![e4b534283554c93ba31ab04d0a14df36871efe910464681304d73aec5cbf9621.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/e4b534283554c93ba31ab04d0a14df36871efe910464681304d73aec5cbf9621.jpg)

![fdc70b4d18984e2877f30b5067a526d1bf8346e6c44411f230a412cd14db1a79.jpg](../icml_results/3780_Balancing%20Learning%20Rates%20Across%20Layers_%20Exact%20Two-Step%20Dynamics%20and%20Optimal%20Scaling%20in%20Linear%20Neural/images/fdc70b4d18984e2877f30b5067a526d1bf8346e6c44411f230a412cd14db1a79.jpg)

<a id="paper-3741"></a>

## EDCO: Dynamic Curriculum Orchestration for Domain-specific Large Language Model Fine-tuning


### Images

![514c8db8c3e6f1e42b845cee7f9aa7db4d172690a555263c714a7bff872a8f5b.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/images/514c8db8c3e6f1e42b845cee7f9aa7db4d172690a555263c714a7bff872a8f5b.jpg)

![a8fd56da8edc362c6fa2c97dd2dfe056bf7f12eb926264d72a1101dae4f0b2ba.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/images/a8fd56da8edc362c6fa2c97dd2dfe056bf7f12eb926264d72a1101dae4f0b2ba.jpg)

![d05befea3e69a04cdacc88b6dbd1d6aaa9ea48f6848dc3e2f4f62ce2d2e1bfcb.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/images/d05befea3e69a04cdacc88b6dbd1d6aaa9ea48f6848dc3e2f4f62ce2d2e1bfcb.jpg)

![da9b0718ed5309b1a8d6d319c25287f60309b9d402a8174ba0e0f12b5667d4df.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/images/da9b0718ed5309b1a8d6d319c25287f60309b9d402a8174ba0e0f12b5667d4df.jpg)

![ea759e4639bb0f11006d8d1839c994d815928a7077dfc6bfe673f7e379b145eb.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/images/ea759e4639bb0f11006d8d1839c994d815928a7077dfc6bfe673f7e379b145eb.jpg)

### Tables

![000467010c6e412caf0b83c8455243fca162873012188891ad28892f5b7f2a1e.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/000467010c6e412caf0b83c8455243fca162873012188891ad28892f5b7f2a1e.jpg)

![12f9c80799ba0f493667d8232ffbbdead3d50ab74a8288c5f8881f63085fc6c4.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/12f9c80799ba0f493667d8232ffbbdead3d50ab74a8288c5f8881f63085fc6c4.jpg)

![232f7e48def62c259670d79cc04914218b416344b1a83d0f3e191ebf3243351b.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/232f7e48def62c259670d79cc04914218b416344b1a83d0f3e191ebf3243351b.jpg)

![32977a4adf0e80f1febdaee873e047304205f5a7b407a1005b552814eeb7770d.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/32977a4adf0e80f1febdaee873e047304205f5a7b407a1005b552814eeb7770d.jpg)

![5336722668bf4ec89a7134ca4caee8501d06800282f7a4899c384053e3fdc535.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/5336722668bf4ec89a7134ca4caee8501d06800282f7a4899c384053e3fdc535.jpg)

![6e4a345dc2fd698c7bf1ad1114ed9de03522e75b4d679abbe828dc2e621e5e23.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/6e4a345dc2fd698c7bf1ad1114ed9de03522e75b4d679abbe828dc2e621e5e23.jpg)

![7787850dc17cd7eef4d9ba68b6243308fc55af6cf6aed043c3e746ded9d46f09.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/7787850dc17cd7eef4d9ba68b6243308fc55af6cf6aed043c3e746ded9d46f09.jpg)

![7c2159f1d3fd429b18e90d2502306c0dcbd6c7bee6eadbac3620b6f08f1ccd13.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/7c2159f1d3fd429b18e90d2502306c0dcbd6c7bee6eadbac3620b6f08f1ccd13.jpg)

![a27fbb5f8d239c0223c906cdf4b429ece31320b7d2576f82667546a04ba6e071.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/a27fbb5f8d239c0223c906cdf4b429ece31320b7d2576f82667546a04ba6e071.jpg)

![b646179d5a3c6f928b7ee96fc04b9f56cb079a71eac6f2e5f6a0325f79c53b3d.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/b646179d5a3c6f928b7ee96fc04b9f56cb079a71eac6f2e5f6a0325f79c53b3d.jpg)

![b90d68590f9584c6c8add08ec5407f6f68a231bed98666ab2275b13057e6afad.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/b90d68590f9584c6c8add08ec5407f6f68a231bed98666ab2275b13057e6afad.jpg)

![c098816d2c86aafc245dc3bac213c572c07c7ae033818f2126ba85c1b1275eaa.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/c098816d2c86aafc245dc3bac213c572c07c7ae033818f2126ba85c1b1275eaa.jpg)

![cdbe3a6352c9167e79a95a82beaa14c8b92f84e4701bca5f1fd40182012f236d.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/cdbe3a6352c9167e79a95a82beaa14c8b92f84e4701bca5f1fd40182012f236d.jpg)

![d9e6019f05f10c4bcf2692d0759ad5f58427a72ecbf57b0316ae6336b64dfcfa.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/d9e6019f05f10c4bcf2692d0759ad5f58427a72ecbf57b0316ae6336b64dfcfa.jpg)

![e03f8b4360a1ba040c8d799cff57ef1c0f948aac89f95a08bc002706755b8f89.jpg](../icml_results/3781_EDCO_%20Dynamic%20Curriculum%20Orchestration%20for%20Domain-specific%20Large%20Language%20Model%20Fine-tuning/tables/e03f8b4360a1ba040c8d799cff57ef1c0f948aac89f95a08bc002706755b8f89.jpg)

<a id="paper-3742"></a>

## Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs


### Images

![35c344b6aa473e3f87e948fdc78d76c0a00c58ce57da341bd087aa9a43a06827.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/35c344b6aa473e3f87e948fdc78d76c0a00c58ce57da341bd087aa9a43a06827.jpg)

![69bb7acb4ca6c7ebf654d67e7d4a0d84118671d32a10fa2058463339258882bf.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/69bb7acb4ca6c7ebf654d67e7d4a0d84118671d32a10fa2058463339258882bf.jpg)

![7e03f5f39919d1ce7264c5dab655389864930fcd05ff50929667489d87a78f87.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/7e03f5f39919d1ce7264c5dab655389864930fcd05ff50929667489d87a78f87.jpg)

![8b7bb6229a97f4859ae0fd3ca91ae1cfe9829cde8fde18046f2af8487c1179e2.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/8b7bb6229a97f4859ae0fd3ca91ae1cfe9829cde8fde18046f2af8487c1179e2.jpg)

![8f33438035ecbddab6897ebebb045ac910cc84b5ca4bff18804d871aa340e482.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/8f33438035ecbddab6897ebebb045ac910cc84b5ca4bff18804d871aa340e482.jpg)

![9c25d91fc1f22afa727154307c63fcea5c785ae254038b63ad1125cc61eeab83.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/9c25d91fc1f22afa727154307c63fcea5c785ae254038b63ad1125cc61eeab83.jpg)

![ccbb7ce3044f5351a4c692788618a45adc928051807955304f5c53bcca70b630.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/ccbb7ce3044f5351a4c692788618a45adc928051807955304f5c53bcca70b630.jpg)

![f082ebad2af15b41b56272f0f28c5a4e0be324b6a176aa62189b4cf574e04619.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/images/f082ebad2af15b41b56272f0f28c5a4e0be324b6a176aa62189b4cf574e04619.jpg)

### Tables

![0b4fd74e5692b8145aeba25dd2b6226dab68ca2e2242360f5ccd590ff157098c.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/0b4fd74e5692b8145aeba25dd2b6226dab68ca2e2242360f5ccd590ff157098c.jpg)

![15cbcc26a51ec9eda3946fa7bce33480ab36d3e55f66ce85cbca875e7a2eb4c2.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/15cbcc26a51ec9eda3946fa7bce33480ab36d3e55f66ce85cbca875e7a2eb4c2.jpg)

![2e303d6c6f3b7698a2f4f2f4d7081f7583bafc087031538cd525db984d05d1ea.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/2e303d6c6f3b7698a2f4f2f4d7081f7583bafc087031538cd525db984d05d1ea.jpg)

![4263c90667a8f91912055cd27fe4682fd9055f4fa799c7617820fd26ceb2634b.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/4263c90667a8f91912055cd27fe4682fd9055f4fa799c7617820fd26ceb2634b.jpg)

![6a5a7e5e9c49d372a7bd0742486392a70ea70cf7ed615f3659f971aebb057f55.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/6a5a7e5e9c49d372a7bd0742486392a70ea70cf7ed615f3659f971aebb057f55.jpg)

![7d658a85a246acc1368f0d786ba0cc415b363719374e46317bc34edfc986fbc1.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/7d658a85a246acc1368f0d786ba0cc415b363719374e46317bc34edfc986fbc1.jpg)

![85fa3116e49c1b1e5b2ce6935507ffb539757ab414bc5cae3f367616216b7716.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/85fa3116e49c1b1e5b2ce6935507ffb539757ab414bc5cae3f367616216b7716.jpg)

![9a8385b51e956ba86aef8d605d27e9395f8cd2b99ec66bfa47bc80d0f11ec679.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/9a8385b51e956ba86aef8d605d27e9395f8cd2b99ec66bfa47bc80d0f11ec679.jpg)

![c322f94dfb0c5280cb4a3fcb51aa407597f70ffceb9b3e5036ef89b848c63795.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/c322f94dfb0c5280cb4a3fcb51aa407597f70ffceb9b3e5036ef89b848c63795.jpg)

![dd2c5f2f4f3a77e7409d459eb82533cdfa0eb750044f8bf4517cae5ba8e07529.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/dd2c5f2f4f3a77e7409d459eb82533cdfa0eb750044f8bf4517cae5ba8e07529.jpg)

![dd9371945ae0ed112aa35114d5d7b1092467f349964c49abb4a7d2100970755c.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/dd9371945ae0ed112aa35114d5d7b1092467f349964c49abb4a7d2100970755c.jpg)

![ef1e45541654a2c68282ff861ec268e0a4d560be9cd02fcf3b05e0ed96eff481.jpg](../icml_results/3782_Deliberate%20Evolution_%20Agentic%20Reasoning%20for%20Sample-Efficient%20Symbolic%20Regression%20with%20LLMs/tables/ef1e45541654a2c68282ff861ec268e0a4d560be9cd02fcf3b05e0ed96eff481.jpg)

<a id="paper-3743"></a>

## Multi-Agent Teams Hold Experts Back


### Images

![0309ffc00aea5254636c28335f16aa1cb286af9e25ebd7cda8027c9e00ac6546.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/0309ffc00aea5254636c28335f16aa1cb286af9e25ebd7cda8027c9e00ac6546.jpg)

![07248785d26e9f44a37f0e1a8c6b7e18ec02227e54588cff48a3d5ed59abab14.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/07248785d26e9f44a37f0e1a8c6b7e18ec02227e54588cff48a3d5ed59abab14.jpg)

![0d323f4c50ff5e92dbae0392b171d24b433fd64c9e98c42de2658b4c46a5f1b2.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/0d323f4c50ff5e92dbae0392b171d24b433fd64c9e98c42de2658b4c46a5f1b2.jpg)

![19e044c27f55d646543a4c18c17ea04c10b116c1b5872d4be954a4bae945e008.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/19e044c27f55d646543a4c18c17ea04c10b116c1b5872d4be954a4bae945e008.jpg)

![1eade543212267482369f4a550cb8accbc4293f6eaae64e6e22da435ec86bfd4.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/1eade543212267482369f4a550cb8accbc4293f6eaae64e6e22da435ec86bfd4.jpg)

![34319b99bec9f8e5010c1d4dfe1c70e443ea05157d363697f0bbfcc47208d4e4.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/34319b99bec9f8e5010c1d4dfe1c70e443ea05157d363697f0bbfcc47208d4e4.jpg)

![497dcc15addf0d002efed72dd34f6e93a76cc54383e02f1cc97ddf69d9cd998d.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/497dcc15addf0d002efed72dd34f6e93a76cc54383e02f1cc97ddf69d9cd998d.jpg)

![4a392955e67125b7d56e0a61b266239faa95c5e77dea89988fcb235f3637d58a.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/4a392955e67125b7d56e0a61b266239faa95c5e77dea89988fcb235f3637d58a.jpg)

![5169461aa67ef38c6654e31a0f110125ccf023fc93a3d97ea792984c2925a661.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/5169461aa67ef38c6654e31a0f110125ccf023fc93a3d97ea792984c2925a661.jpg)

![5dc36318d3e2a68ae887a3eb6fd48de40c3c3304a0b8c4f26853fec6a6cc6b9e.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/5dc36318d3e2a68ae887a3eb6fd48de40c3c3304a0b8c4f26853fec6a6cc6b9e.jpg)

![75c195e43058ce56a5dace1de15a090a513737e71bc63608fc36f12a32d54542.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/75c195e43058ce56a5dace1de15a090a513737e71bc63608fc36f12a32d54542.jpg)

![8ad620dabd8df2ba299b8c9340f2ef5c186e21fb16757f53d13873363bc8f3fd.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/8ad620dabd8df2ba299b8c9340f2ef5c186e21fb16757f53d13873363bc8f3fd.jpg)

![911512baf650c3278343316a136a0eb4b8621770555c5fa0ca2a9120e470490e.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/911512baf650c3278343316a136a0eb4b8621770555c5fa0ca2a9120e470490e.jpg)

![963df444e0a4005f4cca625702a7999cdee5a2b8bca1ead9ef16b990f93af74f.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/963df444e0a4005f4cca625702a7999cdee5a2b8bca1ead9ef16b990f93af74f.jpg)

![e886f49b6291cab63264910225da0975b5563ac02cdef1e529c2b19ad705b843.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/e886f49b6291cab63264910225da0975b5563ac02cdef1e529c2b19ad705b843.jpg)

![fc593a08c0475626f2882acf35eee9a46f9cadf88dceb7a6f9d78d09f6267760.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/images/fc593a08c0475626f2882acf35eee9a46f9cadf88dceb7a6f9d78d09f6267760.jpg)

### Tables

![1355f08fe20eb456ef149166ff5be3206af18a13ae01bab57d1d1d9a020016dd.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/1355f08fe20eb456ef149166ff5be3206af18a13ae01bab57d1d1d9a020016dd.jpg)

![13688d2b3cf071c769720a704193723bc34bff33bb29883188e52f396abade39.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/13688d2b3cf071c769720a704193723bc34bff33bb29883188e52f396abade39.jpg)

![1b97606fb02540121b6440ad4fdf1e66352f8801aee750e9f49dbd650fad03eb.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/1b97606fb02540121b6440ad4fdf1e66352f8801aee750e9f49dbd650fad03eb.jpg)

![41f8d87b2623a1e6e169c7a271cf57e928f6cc652cc22a9194ec7f7ae1d356c8.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/41f8d87b2623a1e6e169c7a271cf57e928f6cc652cc22a9194ec7f7ae1d356c8.jpg)

![5c18b20b2981bee294b63ff0badd52ee529a4cc048030f24c74e3684fffcaa2b.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/5c18b20b2981bee294b63ff0badd52ee529a4cc048030f24c74e3684fffcaa2b.jpg)

![6cb2267dc542347d08b13b6f647be828273b2502e1a13ebd5f1b3e4b9aeb7489.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/6cb2267dc542347d08b13b6f647be828273b2502e1a13ebd5f1b3e4b9aeb7489.jpg)

![70aee01d3c20960754a28d4ecc1e8408c4a57695a8bf95340b72a8ba9190e80b.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/70aee01d3c20960754a28d4ecc1e8408c4a57695a8bf95340b72a8ba9190e80b.jpg)

![cab5fc4b19f5a6a1e91056c970231c42edd10a8c48ef4589c36de239a1475ee9.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/cab5fc4b19f5a6a1e91056c970231c42edd10a8c48ef4589c36de239a1475ee9.jpg)

![d22b9ef6ad2af9e8168867166eaeadc00753069323f7bab689476db945328246.jpg](../icml_results/3783_Multi-Agent%20Teams%20Hold%20Experts%20Back/tables/d22b9ef6ad2af9e8168867166eaeadc00753069323f7bab689476db945328246.jpg)

<a id="paper-3744"></a>

## STARCaster: Spatio-Temporal AutoRegressive Video Diffusion for Identity- and View-Aware Talking Portraits


### Images

![08bfee2627a6fcad7ce926d0186e6508f65308a2eb9694958a44f5497e49db61.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/08bfee2627a6fcad7ce926d0186e6508f65308a2eb9694958a44f5497e49db61.jpg)

![0f1126137dd77757128fd6ba47a971c55bbe80dda582a451c291a5b0e246e16a.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/0f1126137dd77757128fd6ba47a971c55bbe80dda582a451c291a5b0e246e16a.jpg)

![40eab7e983be611377feb0a4e9a1c68ecddc7f934fc9ef88a38f2700ade72cc4.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/40eab7e983be611377feb0a4e9a1c68ecddc7f934fc9ef88a38f2700ade72cc4.jpg)

![49ace8535e76efeeabb9314c98f12679d971424d1df21848b0162fbdc0446184.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/49ace8535e76efeeabb9314c98f12679d971424d1df21848b0162fbdc0446184.jpg)

![4e7a7843af722775cabb8b0216a6174113082d22bab70f24a30eb19ece04bdc2.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/4e7a7843af722775cabb8b0216a6174113082d22bab70f24a30eb19ece04bdc2.jpg)

![6af3ad62ae9d6f5c6d122a3abea541479af9b5837dfe2e6679183c44c074d0f7.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/6af3ad62ae9d6f5c6d122a3abea541479af9b5837dfe2e6679183c44c074d0f7.jpg)

![7d193de7613d26441a54d39d5c4a2c11263064ee368cb092d686eae8ccda202e.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/7d193de7613d26441a54d39d5c4a2c11263064ee368cb092d686eae8ccda202e.jpg)

![7f17c5e3b61f64f0d33c6b7a9db6ddcfdc4cee100a9045272f1da951af1bf788.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/7f17c5e3b61f64f0d33c6b7a9db6ddcfdc4cee100a9045272f1da951af1bf788.jpg)

![8c4be36bc70881fe100525451e6ee06df4a7d74a8bb715480959a27b51e4a2e0.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/8c4be36bc70881fe100525451e6ee06df4a7d74a8bb715480959a27b51e4a2e0.jpg)

![9d4ec3dbe6e909cea6780bb26a37eaef2a7514eee728ead2cc1ebcbfd0adc08d.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/9d4ec3dbe6e909cea6780bb26a37eaef2a7514eee728ead2cc1ebcbfd0adc08d.jpg)

![bb55e83ae76a9d4602faef61cc790cb16cd2a2c9327c962f5db98a309173e6fb.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/bb55e83ae76a9d4602faef61cc790cb16cd2a2c9327c962f5db98a309173e6fb.jpg)

![c14fe483be255eb2eba22f225b7c9452ad03d614c5d96f47a0bea7879c6fce2a.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/c14fe483be255eb2eba22f225b7c9452ad03d614c5d96f47a0bea7879c6fce2a.jpg)

![df6bc5c700647d58029130619cecf04bf8555d1c81fa61f371b7982b591ed3ef.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/df6bc5c700647d58029130619cecf04bf8555d1c81fa61f371b7982b591ed3ef.jpg)

![f0101c978c0c5aa287c6778720942171a37e70350a57901aa2ccfb3588b78b2d.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/images/f0101c978c0c5aa287c6778720942171a37e70350a57901aa2ccfb3588b78b2d.jpg)

### Tables

![481b56b304c0b14ba4e0decf7f83720483a4c5c55c23db99bd71587cc4225a1a.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/tables/481b56b304c0b14ba4e0decf7f83720483a4c5c55c23db99bd71587cc4225a1a.jpg)

![7b0ccc1b737501c9cd5140f498d4ecb856131354df00ebb1f2cea430e174f3ec.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/tables/7b0ccc1b737501c9cd5140f498d4ecb856131354df00ebb1f2cea430e174f3ec.jpg)

![8437b01a77b4d2651917602cc0442378e1e8aa4004af2551fc492313039d8fa2.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/tables/8437b01a77b4d2651917602cc0442378e1e8aa4004af2551fc492313039d8fa2.jpg)

![ea953897a223a9dad2faea9617facde73d3e4369da322c5c15b2de95ba7a911e.jpg](../icml_results/3784_STARCaster_%20Spatio-Temporal%20AutoRegressive%20Video%20Diffusion%20for%20Identity-%20and%20View-Aware%20Talking%20Port/tables/ea953897a223a9dad2faea9617facde73d3e4369da322c5c15b2de95ba7a911e.jpg)

<a id="paper-3745"></a>

## Learnability-Informed Fine-Tuning of Diffusion Language Models


### Images

![0868ef0d99a70cccc54ef20453e731bda64ff5880a9bc7df1714619025afe451.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/0868ef0d99a70cccc54ef20453e731bda64ff5880a9bc7df1714619025afe451.jpg)

![0dcdcfc173360531d87946bdc4a40713246405b24c8c03c4bdb008d7709aeb6d.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/0dcdcfc173360531d87946bdc4a40713246405b24c8c03c4bdb008d7709aeb6d.jpg)

![13e4fe5e59b26e68377d2c24cd08ed24f56727c32da0e476baca48039777e1d5.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/13e4fe5e59b26e68377d2c24cd08ed24f56727c32da0e476baca48039777e1d5.jpg)

![48475ca5c3aa5c92ad1aa5fde3c3725c32230862c662d4a8f82f97af72d5a54f.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/48475ca5c3aa5c92ad1aa5fde3c3725c32230862c662d4a8f82f97af72d5a54f.jpg)

![57dc886057fa49d3d9164629c6d6509d3d8b556213065d8dbb5ab85584911983.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/57dc886057fa49d3d9164629c6d6509d3d8b556213065d8dbb5ab85584911983.jpg)

![627904c735f8fedcdec01b77f145c8ce471befe5c35a47c287495ac76fcc582d.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/627904c735f8fedcdec01b77f145c8ce471befe5c35a47c287495ac76fcc582d.jpg)

![c3201a4e07a388d5d261b664c562cc828dc29417d7da675907bce36002d7b62e.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/c3201a4e07a388d5d261b664c562cc828dc29417d7da675907bce36002d7b62e.jpg)

![d79b09be3cf42df550ceac7906486116f76b01c7c16f0f3e53df1712f0a5ab0e.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/d79b09be3cf42df550ceac7906486116f76b01c7c16f0f3e53df1712f0a5ab0e.jpg)

![da05ce41dda87dc6813ec27f7d00d1553c2cdb71b52187778570acebe7904125.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/da05ce41dda87dc6813ec27f7d00d1553c2cdb71b52187778570acebe7904125.jpg)

![f2ec4a4f24893d7e6eaa78ff974f112cade8903c1d7974e8a23b7c53f3c2bbe4.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/images/f2ec4a4f24893d7e6eaa78ff974f112cade8903c1d7974e8a23b7c53f3c2bbe4.jpg)

### Tables

![0db7f68c87c868e0f5c92b0ddd61e9545b0757402bfaaf42de5e5d3de666746e.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/0db7f68c87c868e0f5c92b0ddd61e9545b0757402bfaaf42de5e5d3de666746e.jpg)

![15717d6910226ad19f6eabf0116c9dadf9d27d117221de7d596b05b608c43ab2.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/15717d6910226ad19f6eabf0116c9dadf9d27d117221de7d596b05b608c43ab2.jpg)

![1737c4310c652b1fd6899ac5254bae3650fa83c08b59018a378e8006df668722.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/1737c4310c652b1fd6899ac5254bae3650fa83c08b59018a378e8006df668722.jpg)

![1a36f59c90d258103ac3a37278e9ffb3a7d627333338bc62db618ea39264221d.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/1a36f59c90d258103ac3a37278e9ffb3a7d627333338bc62db618ea39264221d.jpg)

![21622e1f81b4585c8cb0b56dc4d3000990cf9705d7083cd9ac3c5cb49283a9af.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/21622e1f81b4585c8cb0b56dc4d3000990cf9705d7083cd9ac3c5cb49283a9af.jpg)

![4734d25d33db45e4e96a599357d256e08c3bcce1f02fbb7652dfd025fd18e146.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/4734d25d33db45e4e96a599357d256e08c3bcce1f02fbb7652dfd025fd18e146.jpg)

![4ff7699a037ff9158deed51724f8a480b5a603d815ffc0a8fde663f86879a852.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/4ff7699a037ff9158deed51724f8a480b5a603d815ffc0a8fde663f86879a852.jpg)

![76acfa8349f664b6b8c254cf90479ddeaeb8221067667017949efa843e9c87d2.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/76acfa8349f664b6b8c254cf90479ddeaeb8221067667017949efa843e9c87d2.jpg)

![b575f6a033aa65fbf767e8cf81345f0a964e9aa92e04d31161d97a6d64eda523.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/b575f6a033aa65fbf767e8cf81345f0a964e9aa92e04d31161d97a6d64eda523.jpg)

![c54da8d2cd2fff39c75fcca4d5d183af83130ec439d554d660e87bd8e2e21da6.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/c54da8d2cd2fff39c75fcca4d5d183af83130ec439d554d660e87bd8e2e21da6.jpg)

![cbc6ff89091e0d927b5c657e63d4d33228848a09ba2b148ba23a6f5338ee8c96.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/cbc6ff89091e0d927b5c657e63d4d33228848a09ba2b148ba23a6f5338ee8c96.jpg)

![e7cd5d746bcae223a0b6fd9192022ee8c88ca40a7c3349fd9d2eb56570f2de52.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/e7cd5d746bcae223a0b6fd9192022ee8c88ca40a7c3349fd9d2eb56570f2de52.jpg)

![f1517c8cf3fc138d01760c4c4e8ae15c0c68a2ee23ca99c752d09d80b7d65842.jpg](../icml_results/3785_Learnability-Informed%20Fine-Tuning%20of%20Diffusion%20Language%20Models/tables/f1517c8cf3fc138d01760c4c4e8ae15c0c68a2ee23ca99c752d09d80b7d65842.jpg)

<a id="paper-3746"></a>

## Online Social Welfare Function-based Resource Allocation


### Images

![03a1dbd568f1ccc16bbc8087bea5847c634d71c47c99b7e8ba2e598aca3b3c0e.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/03a1dbd568f1ccc16bbc8087bea5847c634d71c47c99b7e8ba2e598aca3b3c0e.jpg)

![3c7bcd48ece8d5244bd29599980199544abfc21b0ae2397a79022e8abc0526a2.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/3c7bcd48ece8d5244bd29599980199544abfc21b0ae2397a79022e8abc0526a2.jpg)

![577a03ed69705e483047d953ba93bb741792aa6864775fce721f46af4a9d89b7.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/577a03ed69705e483047d953ba93bb741792aa6864775fce721f46af4a9d89b7.jpg)

![7946b98366c949b30d72a9da22eeda921e7e0867dc1007acacfb13808ff8490e.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/7946b98366c949b30d72a9da22eeda921e7e0867dc1007acacfb13808ff8490e.jpg)

![93624eeba63d9d7ad4243fdb30d82e363c44266e21b95375fced957c210f8fe8.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/93624eeba63d9d7ad4243fdb30d82e363c44266e21b95375fced957c210f8fe8.jpg)

![a6ff9e2363df67190b4042d40b812d7c05dbd43d818e847cc41b3b011dfcb872.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/a6ff9e2363df67190b4042d40b812d7c05dbd43d818e847cc41b3b011dfcb872.jpg)

![df5dfd259fe65eadd2628e89b37a1377f2a2ce839a8f753b689492ebcf938a3b.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/images/df5dfd259fe65eadd2628e89b37a1377f2a2ce839a8f753b689492ebcf938a3b.jpg)

### Tables

![acb889a2827ef86638da7608e0ada6347f99130a052fe538eaae46c9c35f9d51.jpg](../icml_results/3786_Online%20Social%20Welfare%20Function-based%20Resource%20Allocation/tables/acb889a2827ef86638da7608e0ada6347f99130a052fe538eaae46c9c35f9d51.jpg)

<a id="paper-3747"></a>

## TabularBERT: Binning-Based Self-Supervised Learning for Tabular Representation


### Images

![1204cb69df738355c53580c7e00a30db9e6307e49bce545d5d41991a2621ffb7.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/images/1204cb69df738355c53580c7e00a30db9e6307e49bce545d5d41991a2621ffb7.jpg)

![4b3b144249051b5eec741678b52da0e82d7a841c4c69836cd79a714aa7e45ac5.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/images/4b3b144249051b5eec741678b52da0e82d7a841c4c69836cd79a714aa7e45ac5.jpg)

![559b08fca15cbc1196078b076a27dad987dc36977f5635a134281256b700be6f.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/images/559b08fca15cbc1196078b076a27dad987dc36977f5635a134281256b700be6f.jpg)

![7412a814d2d7e5986e70f4fab469f2e2ab9c310aa1a01703cc052270cc8063b4.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/images/7412a814d2d7e5986e70f4fab469f2e2ab9c310aa1a01703cc052270cc8063b4.jpg)

![a194304f866fc6a514a9c47592dd703c745d423b737b261b1350df00e4fc15da.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/images/a194304f866fc6a514a9c47592dd703c745d423b737b261b1350df00e4fc15da.jpg)

![d5e1678f795f96f03f87ebd58ad9ecd6487a2ce327b27407d59ace0b469518ef.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/images/d5e1678f795f96f03f87ebd58ad9ecd6487a2ce327b27407d59ace0b469518ef.jpg)

### Tables

![432fa43a7f08faa2befed6a06435c4405a6b4d4181da1610bfcac490afdf6787.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/432fa43a7f08faa2befed6a06435c4405a6b4d4181da1610bfcac490afdf6787.jpg)

![6326ba521daac5accdb1645322e4c2e58b95237b6b61a9a7af1cc5b127215ae6.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/6326ba521daac5accdb1645322e4c2e58b95237b6b61a9a7af1cc5b127215ae6.jpg)

![64f22a265dd7c7a250a45adbcc993408965d69696a8b4a839441916d4f7c1d57.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/64f22a265dd7c7a250a45adbcc993408965d69696a8b4a839441916d4f7c1d57.jpg)

![6c5536be2620404afff9d197391f12e8bdef4ec94f23833e2f0b7d9f41d85a2c.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/6c5536be2620404afff9d197391f12e8bdef4ec94f23833e2f0b7d9f41d85a2c.jpg)

![82234e4082a1b0cc6d709959725c9542e3c12e2e0f7a0519ab9bcd42c274dedb.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/82234e4082a1b0cc6d709959725c9542e3c12e2e0f7a0519ab9bcd42c274dedb.jpg)

![888e08a2465f10910b79f4fb26d2e176e4b98763affef610ece864c7ae8869f6.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/888e08a2465f10910b79f4fb26d2e176e4b98763affef610ece864c7ae8869f6.jpg)

![91d6b08c4fe7ff7dd77e81bf3ad781c5918e3c763e7347c7f7f70eb26b0a8865.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/91d6b08c4fe7ff7dd77e81bf3ad781c5918e3c763e7347c7f7f70eb26b0a8865.jpg)

![9e3bba6fca2255dcfbf4123de1ee15efb9bfbe5af5877a9f2f6029b01d326e36.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/9e3bba6fca2255dcfbf4123de1ee15efb9bfbe5af5877a9f2f6029b01d326e36.jpg)

![a2d55d9196b2b71cd6751da59c2ab52f485efdd7201eb295d5112bdb4639fac0.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/a2d55d9196b2b71cd6751da59c2ab52f485efdd7201eb295d5112bdb4639fac0.jpg)

![a5dfc33ceb16b4436c45b22b9517904ae893cba3d554917c9035980eaafa283c.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/a5dfc33ceb16b4436c45b22b9517904ae893cba3d554917c9035980eaafa283c.jpg)

![bd8d8e386c0dd6abd41085789404d1cc68783d06988174e2c294f3f56200f06c.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/bd8d8e386c0dd6abd41085789404d1cc68783d06988174e2c294f3f56200f06c.jpg)

![f539f9f07923e63d1c5f3eb4434551bce61fcb7ecb0a1d1db57eac8e78677351.jpg](../icml_results/3787_TabularBERT_%20Binning-Based%20Self-Supervised%20Learning%20for%20Tabular%20Representation/tables/f539f9f07923e63d1c5f3eb4434551bce61fcb7ecb0a1d1db57eac8e78677351.jpg)

<a id="paper-3748"></a>

## Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach


### Images

![1b59f2a8690503a1c7b549c63910eed2c76c1b606facfcffde3610cd4adf8b3d.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/1b59f2a8690503a1c7b549c63910eed2c76c1b606facfcffde3610cd4adf8b3d.jpg)

![1d4086d928e2776db2b2042ce7da702e1fe43e5f13757cb1fa59069f2ca44e19.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/1d4086d928e2776db2b2042ce7da702e1fe43e5f13757cb1fa59069f2ca44e19.jpg)

![3f0431771bf162460ac8ff91cf86a8b3989c963d2c0ea42625634aae3f57682c.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/3f0431771bf162460ac8ff91cf86a8b3989c963d2c0ea42625634aae3f57682c.jpg)

![45389363b1c626943e2654d981e74aebac393991d49d93914daf653aa21f5df4.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/45389363b1c626943e2654d981e74aebac393991d49d93914daf653aa21f5df4.jpg)

![4c7da714804c478b3e63480c1cccc2424f050fb669d33da0a059b9a992bc9fac.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/4c7da714804c478b3e63480c1cccc2424f050fb669d33da0a059b9a992bc9fac.jpg)

![51b76f283648db22c7010a3cc028b4a8fb3cc302b986ed38cec3ed451e3e71bc.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/51b76f283648db22c7010a3cc028b4a8fb3cc302b986ed38cec3ed451e3e71bc.jpg)

![53dbea767f32197f943b98cfd8eb72ac24b51700c15357db30032ffc775bdfa6.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/53dbea767f32197f943b98cfd8eb72ac24b51700c15357db30032ffc775bdfa6.jpg)

![571374c014fa956bbde91502708c8a5f5ad32b2e7a3cef2b8f6680a75d4cba3e.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/571374c014fa956bbde91502708c8a5f5ad32b2e7a3cef2b8f6680a75d4cba3e.jpg)

![65d0ca5b4703a66e43cb9e822511bbbb892c62c3309a56ff244c8b590e0f98b3.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/65d0ca5b4703a66e43cb9e822511bbbb892c62c3309a56ff244c8b590e0f98b3.jpg)

![69b7a2e9db0835dfc574ba7069fbefe21082e833578d0db02c6c488ac2e79c94.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/69b7a2e9db0835dfc574ba7069fbefe21082e833578d0db02c6c488ac2e79c94.jpg)

![6a7f2bbd12cc1702ba4f0344d5a89fed1356b7b8eb88f1c1b0122c22c1cb2b8e.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/6a7f2bbd12cc1702ba4f0344d5a89fed1356b7b8eb88f1c1b0122c22c1cb2b8e.jpg)

![766487d0fd9a663f889ff712bbf4ccb7c7075dcf0dde2007dcebc0407ddb6ac3.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/766487d0fd9a663f889ff712bbf4ccb7c7075dcf0dde2007dcebc0407ddb6ac3.jpg)

![7a4ce5308fd1badb031592b355d8b87616c7d14b3f5b2db1a04804f88d991754.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/7a4ce5308fd1badb031592b355d8b87616c7d14b3f5b2db1a04804f88d991754.jpg)

![7d76cd33a067df9429808521df91e097f217bbdd7a05f93ffcf61d4a0aa3e73b.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/7d76cd33a067df9429808521df91e097f217bbdd7a05f93ffcf61d4a0aa3e73b.jpg)

![9e7a9359f157c5240c2dd442742762c40635036cd6d4af8a66e59ba4e6448a82.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/9e7a9359f157c5240c2dd442742762c40635036cd6d4af8a66e59ba4e6448a82.jpg)

![a1c8a8ebb622d12f0a38ea031722ceb21ef0d9f5fed007ab5c3e609d79c7b311.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/a1c8a8ebb622d12f0a38ea031722ceb21ef0d9f5fed007ab5c3e609d79c7b311.jpg)

![a5fb10b15301f6f4246e87ccd7ca6962a4224795e1d004c854ef62f8cdd0b0cf.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/a5fb10b15301f6f4246e87ccd7ca6962a4224795e1d004c854ef62f8cdd0b0cf.jpg)

![a86a788868ae86445b706fb5589ba6a18bf95af1f80bee7d71d45f9f3f7b3a57.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/a86a788868ae86445b706fb5589ba6a18bf95af1f80bee7d71d45f9f3f7b3a57.jpg)

![aaece1d882ba968dd2f8032e6d4dbc60285a112d650849dce6692133f42626c8.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/aaece1d882ba968dd2f8032e6d4dbc60285a112d650849dce6692133f42626c8.jpg)

![abc5935d3ed0403107a0273d95ee066edfabfd140a9bfdb1def6741f265c391b.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/abc5935d3ed0403107a0273d95ee066edfabfd140a9bfdb1def6741f265c391b.jpg)

![b1b87df7e0ceb1a924151fad6a32a44a8b8a05f4836faf82c0964aead71458de.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/b1b87df7e0ceb1a924151fad6a32a44a8b8a05f4836faf82c0964aead71458de.jpg)

![b4d617f7c6186e5ea9d910ddf5bf8e7f6ca88bee831b1e1553ccc03a1ebfccfb.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/b4d617f7c6186e5ea9d910ddf5bf8e7f6ca88bee831b1e1553ccc03a1ebfccfb.jpg)

![b73ae45cf44cfc26b3b59b15a7ed86d227f3b05b9fa27e232c7b300a021d2592.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/b73ae45cf44cfc26b3b59b15a7ed86d227f3b05b9fa27e232c7b300a021d2592.jpg)

![d61fdb4014405e8f7e31edd1ea26da51f1c74b6a41d4b53d65c4e8147e77867e.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/d61fdb4014405e8f7e31edd1ea26da51f1c74b6a41d4b53d65c4e8147e77867e.jpg)

![d8ebb66a7eeb7dfcbe66d0293b143506c5d2ed58f1260475b38f97e2cad0a038.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/d8ebb66a7eeb7dfcbe66d0293b143506c5d2ed58f1260475b38f97e2cad0a038.jpg)

![d9425b66749437529efd5f0a86906b8b80f39d19728ccfbfa8d487661a99349f.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/d9425b66749437529efd5f0a86906b8b80f39d19728ccfbfa8d487661a99349f.jpg)

![e43c569269826b7fe5ae9d0a00bd0c466ef042242a6e3a9203e86c7f0d9957ec.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/e43c569269826b7fe5ae9d0a00bd0c466ef042242a6e3a9203e86c7f0d9957ec.jpg)

![f36971512816539df41a5e2b80d2d2e9e35000810faca74cddb39bba9f569dba.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/f36971512816539df41a5e2b80d2d2e9e35000810faca74cddb39bba9f569dba.jpg)

![feaa2f725698717565a90a51c2c83c7526dc05dc97a3af98bb6debd24d3f19a7.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/images/feaa2f725698717565a90a51c2c83c7526dc05dc97a3af98bb6debd24d3f19a7.jpg)

### Tables

![100285b161cb794303d08f85e4fe2aa15e7baa910a70498de2a0ba6fc5e862f5.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/100285b161cb794303d08f85e4fe2aa15e7baa910a70498de2a0ba6fc5e862f5.jpg)

![1fb3e5c3f78824e2a9dc0cce02747a61a89de933e8913995cf66234fd0ba19bd.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/1fb3e5c3f78824e2a9dc0cce02747a61a89de933e8913995cf66234fd0ba19bd.jpg)

![217d95578454be117973954a6b4ae5d989676534dc0eeebbd3af4f362be9c8c4.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/217d95578454be117973954a6b4ae5d989676534dc0eeebbd3af4f362be9c8c4.jpg)

![218a80288a78d3e5de6b2d5e6ee73abca3e9531e9a2d1b41081ec7beda526b01.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/218a80288a78d3e5de6b2d5e6ee73abca3e9531e9a2d1b41081ec7beda526b01.jpg)

![2ede0de70ea3dfcd651fed2e61ac5de1e79d336b6f64c7af982721a5582dfa29.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/2ede0de70ea3dfcd651fed2e61ac5de1e79d336b6f64c7af982721a5582dfa29.jpg)

![38762a82f88810f0adef43473e15964b79dd17ed28e69623459669d9a8e2ffbd.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/38762a82f88810f0adef43473e15964b79dd17ed28e69623459669d9a8e2ffbd.jpg)

![573419691b35f558303b6bb5d22af13b5da354d543d6d68f8358887d9c045111.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/573419691b35f558303b6bb5d22af13b5da354d543d6d68f8358887d9c045111.jpg)

![579f8fbe4593eeb9082abc701cbfa0a7ffed66241a23ebf9fb6820dbd0b17c0d.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/579f8fbe4593eeb9082abc701cbfa0a7ffed66241a23ebf9fb6820dbd0b17c0d.jpg)

![676a1cc9582fbd88038443bc3d0f6239697e809ebaa8b0c751207484b6f88030.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/676a1cc9582fbd88038443bc3d0f6239697e809ebaa8b0c751207484b6f88030.jpg)

![6eefe5eb9b0322b48a89496ffb98b73b54caad10fac3b3c3414f15dd32d5d42d.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/6eefe5eb9b0322b48a89496ffb98b73b54caad10fac3b3c3414f15dd32d5d42d.jpg)

![6f7b7dd699fa532b18cc080948b3008b5a28a59f28121db5c5aae64ad0e6b169.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/6f7b7dd699fa532b18cc080948b3008b5a28a59f28121db5c5aae64ad0e6b169.jpg)

![7092f8f401a327ba4b28aac1e10a6bfa0178b88b79d70461728bb3b6da9cb8e4.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/7092f8f401a327ba4b28aac1e10a6bfa0178b88b79d70461728bb3b6da9cb8e4.jpg)

![7641939bb8a4099e2dbf98c7013885e8b624bda8b785d1c7fcd5c7f90b8e3e14.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/7641939bb8a4099e2dbf98c7013885e8b624bda8b785d1c7fcd5c7f90b8e3e14.jpg)

![b82dfcd0d3b35461e1554d2f101dfa6fd53b37b6056b638e830240493041cdcb.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/b82dfcd0d3b35461e1554d2f101dfa6fd53b37b6056b638e830240493041cdcb.jpg)

![efcd9e1fe86606e958a68c50fa34e23f76debfe48a1c7bf1a55fa54096a24526.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/efcd9e1fe86606e958a68c50fa34e23f76debfe48a1c7bf1a55fa54096a24526.jpg)

![f008d550fce59c40e4b8718ff1fe87465d12d1827047dcbe325cccbc3e824f0d.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/f008d550fce59c40e4b8718ff1fe87465d12d1827047dcbe325cccbc3e824f0d.jpg)

![f0ed511b95bdca1ac3ac2fc7f42d9d3558176e20f98a6689b5af3def16439c74.jpg](../icml_results/3788_Post-Training%20LLMs%20as%20Better%20Decision-Making%20Agents_%20A%20Regret-Minimization%20Approach/tables/f0ed511b95bdca1ac3ac2fc7f42d9d3558176e20f98a6689b5af3def16439c74.jpg)

<a id="paper-3749"></a>

## Mitigating Perceptual Judgment Bias in Multimodal LLM-as-a-Judge via Perceptual Perturbation and Reward Modeling


### Images

![079b86af485193d890a331c14ce55bf18e259113eec33b4a7d2032265faae603.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/079b86af485193d890a331c14ce55bf18e259113eec33b4a7d2032265faae603.jpg)

![19c297b133606eca26934f7bd5fa88f80249d6cd45cbc54315be9b7c6de3019b.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/19c297b133606eca26934f7bd5fa88f80249d6cd45cbc54315be9b7c6de3019b.jpg)

![1c517959f8b31f490c5c63beb41bd9bf3b3035b8336e89a351d33fb7b88190a8.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/1c517959f8b31f490c5c63beb41bd9bf3b3035b8336e89a351d33fb7b88190a8.jpg)

![20a78ed1709be34378e49d9de858a8202fea4d9534b466819a5b70d84b595541.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/20a78ed1709be34378e49d9de858a8202fea4d9534b466819a5b70d84b595541.jpg)

![42e42bd1caebef7bc1f2b111f34e73ac3c33d195f134a565b24e2281a80ad01e.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/42e42bd1caebef7bc1f2b111f34e73ac3c33d195f134a565b24e2281a80ad01e.jpg)

![514f8618ec3aa9af9283fd7457728692807a990310863bb9800ea91dc3a89860.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/514f8618ec3aa9af9283fd7457728692807a990310863bb9800ea91dc3a89860.jpg)

![7b9540b556f53d5a1baeef20605b74198fe7d291746e2ca86b422966c758937b.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/7b9540b556f53d5a1baeef20605b74198fe7d291746e2ca86b422966c758937b.jpg)

![9a09595b3a6ea11d1f3623aa538a71acd630928636d0053adf218c64195b5d06.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/9a09595b3a6ea11d1f3623aa538a71acd630928636d0053adf218c64195b5d06.jpg)

![9fc4d031b2cb121ee725429e9aa82c32f1a171ae97480e52de7334caa872b050.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/9fc4d031b2cb121ee725429e9aa82c32f1a171ae97480e52de7334caa872b050.jpg)

![a381d945b16ca16b90a2140d673ae145ec006a6855875e7121d6c67951043fcb.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/a381d945b16ca16b90a2140d673ae145ec006a6855875e7121d6c67951043fcb.jpg)

![adb19c26e78d77a7218277e74b73efd778efb0067501f0477b7fdccbda4e9e52.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/adb19c26e78d77a7218277e74b73efd778efb0067501f0477b7fdccbda4e9e52.jpg)

![c211ac10884863ea8dacc9148e21f4cae6c3488b2db6b4cd71f96b088fc76894.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/c211ac10884863ea8dacc9148e21f4cae6c3488b2db6b4cd71f96b088fc76894.jpg)

![c2910b042a5407bdb56337dcbb992eb63fc1cfd59cc8ae78734483150bb59c50.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/c2910b042a5407bdb56337dcbb992eb63fc1cfd59cc8ae78734483150bb59c50.jpg)

![d0c2b3128a9cf3736e869085a1e89b66f8496f36602e376a00773cc4ce2aee24.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/d0c2b3128a9cf3736e869085a1e89b66f8496f36602e376a00773cc4ce2aee24.jpg)

![d47707f2544203cd2204c8fc1bc21bc23853e48b287b830653561bb67f8136bf.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/d47707f2544203cd2204c8fc1bc21bc23853e48b287b830653561bb67f8136bf.jpg)

![f7834b0bcd707e7f952d7e594bf8f8735649db2081da68901ace068780d6b5b9.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/images/f7834b0bcd707e7f952d7e594bf8f8735649db2081da68901ace068780d6b5b9.jpg)

### Tables

![1005e56faf5f9c285dd17605489147336d98a542b7a7e30279d254461bf967b5.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/1005e56faf5f9c285dd17605489147336d98a542b7a7e30279d254461bf967b5.jpg)

![1bbb15d6f4f5b3d95778608b96952db8d42d2fddb4438b865558377e5ed79d2b.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/1bbb15d6f4f5b3d95778608b96952db8d42d2fddb4438b865558377e5ed79d2b.jpg)

![26ff6af675e39a8d8291347a8cec4fa4ee09d56de389f6c652c66fde80be11cd.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/26ff6af675e39a8d8291347a8cec4fa4ee09d56de389f6c652c66fde80be11cd.jpg)

![2db2fce9b222c41be35c83d028b7f213b3ba583b3254f13f09fedb28005c1ec4.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/2db2fce9b222c41be35c83d028b7f213b3ba583b3254f13f09fedb28005c1ec4.jpg)

![3506236fc2a72466c339908f5cc63ddb69738f0152da62178d01aefadce6d0aa.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/3506236fc2a72466c339908f5cc63ddb69738f0152da62178d01aefadce6d0aa.jpg)

![519db38e05571d7dd8346af86ed03bbc1aa01330ed65d7058158699790c753e9.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/519db38e05571d7dd8346af86ed03bbc1aa01330ed65d7058158699790c753e9.jpg)

![642e5a9f02f3347df8ee69a356999598fca5f99d1b6b2604573b0933d1baea0e.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/642e5a9f02f3347df8ee69a356999598fca5f99d1b6b2604573b0933d1baea0e.jpg)

![b1b22685dbe46e889c1ec721056a27909806472aa55e4e65977dd6954dea0cd8.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/b1b22685dbe46e889c1ec721056a27909806472aa55e4e65977dd6954dea0cd8.jpg)

![d3fba79e28def10ca61bfe7ea13a7c7e5532904e8c45fddc9521cb75c522a550.jpg](../icml_results/3789_Mitigating%20Perceptual%20Judgment%20Bias%20in%20Multimodal%20LLM-as-a-Judge%20via%20Perceptual%20Perturbation%20and%20Rew/tables/d3fba79e28def10ca61bfe7ea13a7c7e5532904e8c45fddc9521cb75c522a550.jpg)

<a id="paper-3750"></a>

## Convergent World Representations and Divergent Tasks


### Images

![035e66480ffff2f46bd7bdaaa8996452240e1b683fe21aac7118c945740ead3a.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/035e66480ffff2f46bd7bdaaa8996452240e1b683fe21aac7118c945740ead3a.jpg)

![07cc34dbcc11d1c94fc109eb0c1abfde9ea70667df27d763db0989a13e7bb999.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/07cc34dbcc11d1c94fc109eb0c1abfde9ea70667df27d763db0989a13e7bb999.jpg)

![1afaa5912fef748b325680c19c787b8855eb13fad207e17d8a1f4210752f1d3e.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/1afaa5912fef748b325680c19c787b8855eb13fad207e17d8a1f4210752f1d3e.jpg)

![267925d4148ec122d5c7fe814a021310fcc8199f05ab4321f97825c90d825063.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/267925d4148ec122d5c7fe814a021310fcc8199f05ab4321f97825c90d825063.jpg)

![26fd4b7add5315636deb84c3acf83fd2d5079d434241425c319192d31a30d96b.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/26fd4b7add5315636deb84c3acf83fd2d5079d434241425c319192d31a30d96b.jpg)

![3d8529390668d557f7390a17170161de5515971c9ebddec3c1b56b9c8586b104.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/3d8529390668d557f7390a17170161de5515971c9ebddec3c1b56b9c8586b104.jpg)

![5589d40ba4540ab97ac4bfce49a4c484e3af6c08c18a4d96b4d61c6619dcb884.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/5589d40ba4540ab97ac4bfce49a4c484e3af6c08c18a4d96b4d61c6619dcb884.jpg)

![5a78f3a50a3061ca5fb3e1dffd17605c9dfe758df6c3b9587e7d286f4f0e76b8.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/5a78f3a50a3061ca5fb3e1dffd17605c9dfe758df6c3b9587e7d286f4f0e76b8.jpg)

![62e849974837a281bc3dd1d8f0f25ba6a4ca361e220126dd93af672fc736eeb1.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/62e849974837a281bc3dd1d8f0f25ba6a4ca361e220126dd93af672fc736eeb1.jpg)

![7b85a7ae1458daab35b97919ad201b6639fcea63282ad5adb360bf6d13aa8911.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/7b85a7ae1458daab35b97919ad201b6639fcea63282ad5adb360bf6d13aa8911.jpg)

![7eddc846e8f1c180a49673ccdb2419bd305696448d71738b9955f9acbf0d7521.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/7eddc846e8f1c180a49673ccdb2419bd305696448d71738b9955f9acbf0d7521.jpg)

![8e3460ab1452cbe51f87d137f4eed685368b0f2602fd457d73924722d854b0de.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/8e3460ab1452cbe51f87d137f4eed685368b0f2602fd457d73924722d854b0de.jpg)

![92792c1dae40b96bb88a5f875e88349ad140df97efc26d09a6fbcbfeba9a9c4d.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/92792c1dae40b96bb88a5f875e88349ad140df97efc26d09a6fbcbfeba9a9c4d.jpg)

![ab38b096c03f148fb2413147947fe01ef9b47be6fd67ae5c7a24498b38220444.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/ab38b096c03f148fb2413147947fe01ef9b47be6fd67ae5c7a24498b38220444.jpg)

![b8e4fe3b7fe6bf8c8dc1f97f7552562b9c0e44bbc2571caaef0520e91b4c17a4.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/b8e4fe3b7fe6bf8c8dc1f97f7552562b9c0e44bbc2571caaef0520e91b4c17a4.jpg)

![bbf4aabfb4438e4fb39d9d64eccd6e8b4bccb2a4cf45cc704c8b04569d0864f0.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/bbf4aabfb4438e4fb39d9d64eccd6e8b4bccb2a4cf45cc704c8b04569d0864f0.jpg)

![bd07c7ecc4201e0c9142088d6f8f2f2d3c457034ab60c7094a4911010cf7be03.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/bd07c7ecc4201e0c9142088d6f8f2f2d3c457034ab60c7094a4911010cf7be03.jpg)

![d40f7ac99d14249473f8d3ed634e92ec1e939d87cfc7db498197aff618397600.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/d40f7ac99d14249473f8d3ed634e92ec1e939d87cfc7db498197aff618397600.jpg)

![d5419072929aca91c1871281227a23e8fad60087a54ececd3bee3bad1b327fe4.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/d5419072929aca91c1871281227a23e8fad60087a54ececd3bee3bad1b327fe4.jpg)

![ed53c99eb44eacba34a13d3c94ac2cf18829ca388470bcb848e5fef5a493f6a4.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/images/ed53c99eb44eacba34a13d3c94ac2cf18829ca388470bcb848e5fef5a493f6a4.jpg)

### Tables

![33930f04a407fcd022f9800c4e8d6a5b85aa03392a653fd1a4565ad63b64b995.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/tables/33930f04a407fcd022f9800c4e8d6a5b85aa03392a653fd1a4565ad63b64b995.jpg)

![84031e231d292268a76b964e2ff0d38667edb39b846002c6c9c7a93f7cd00676.jpg](../icml_results/3790_Convergent%20World%20Representations%20and%20Divergent%20Tasks/tables/84031e231d292268a76b964e2ff0d38667edb39b846002c6c9c7a93f7cd00676.jpg)

<a id="paper-3751"></a>

## Sharpness-Aware Minimization Can Hallucinate Minimizers


### Images

![17206645b527f81a411d97a1fbc447ca6bc1256bb2713ece2ece44f7e2740597.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/17206645b527f81a411d97a1fbc447ca6bc1256bb2713ece2ece44f7e2740597.jpg)

![2bfa50ce0738c8c171bf4c7779d13d279c4bdd8cd65e1c41ccf1da785b8fd8ef.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/2bfa50ce0738c8c171bf4c7779d13d279c4bdd8cd65e1c41ccf1da785b8fd8ef.jpg)

![4a633a2c83c247f1b87c8f5bcbe23af4ac9e0871baf81e1dd50e96ce59ba403c.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/4a633a2c83c247f1b87c8f5bcbe23af4ac9e0871baf81e1dd50e96ce59ba403c.jpg)

![598e84f9a754eced460cb2e43f9acdb3cca24cc247ac07a0efe2c816edad5512.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/598e84f9a754eced460cb2e43f9acdb3cca24cc247ac07a0efe2c816edad5512.jpg)

![6dc083a381010ecf2c8f4a67c09b96bc7c854ec15578a13a9803194c684782f3.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/6dc083a381010ecf2c8f4a67c09b96bc7c854ec15578a13a9803194c684782f3.jpg)

![b90a7e2a863c4bdea168dcc81a5dc42c8473e68eae126fe05507445436096ff9.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/b90a7e2a863c4bdea168dcc81a5dc42c8473e68eae126fe05507445436096ff9.jpg)

![bd686e4c6ad36daee7f6dc7c09085ed54fd107973a4522ba1faae7f5495e6300.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/bd686e4c6ad36daee7f6dc7c09085ed54fd107973a4522ba1faae7f5495e6300.jpg)

![bf51638516c46f024fdcf1fb5041e4415b5c96566f85715190d954dcb29cac8a.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/bf51638516c46f024fdcf1fb5041e4415b5c96566f85715190d954dcb29cac8a.jpg)

![c502e4fd62ced8e8c50c0d2682385dd74221e93ac3a63dfdeceac55c8b16603c.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/c502e4fd62ced8e8c50c0d2682385dd74221e93ac3a63dfdeceac55c8b16603c.jpg)

![d3ec0ba4bdce452309a5775f52d9b17deabe3fa74beb52e3f1d4ff1c652e0bcf.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/d3ec0ba4bdce452309a5775f52d9b17deabe3fa74beb52e3f1d4ff1c652e0bcf.jpg)

![eb2dffca6a0ca23936fd2f1465f2422a942f036c7165e75bd6b463ad22901a56.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/eb2dffca6a0ca23936fd2f1465f2422a942f036c7165e75bd6b463ad22901a56.jpg)

![ed68fae64347d6b5876d90570245e139cef2648a84eb82af1441852ac8ed4206.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/images/ed68fae64347d6b5876d90570245e139cef2648a84eb82af1441852ac8ed4206.jpg)

### Tables

![a1cc387268a894befe41058ab2a61f7fac50c5d0c4ce946836948d6471dbe793.jpg](../icml_results/3791_Sharpness-Aware%20Minimization%20Can%20Hallucinate%20Minimizers/tables/a1cc387268a894befe41058ab2a61f7fac50c5d0c4ce946836948d6471dbe793.jpg)

<a id="paper-3752"></a>

## Q-Delta: Beyond Key–Value Associative State Evolution


### Images

![1791da31d96ba6b5ed7c733af38342209e1746edf2c2db61e3b2aeccc88b282e.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/images/1791da31d96ba6b5ed7c733af38342209e1746edf2c2db61e3b2aeccc88b282e.jpg)

![3368c81ab22d454faae8ca583de665803ea7fefd65e94900a2648ebe851ef65f.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/images/3368c81ab22d454faae8ca583de665803ea7fefd65e94900a2648ebe851ef65f.jpg)

![3b9eafca8aab71a0761f38bfa35fdfbcb3a7fdff30eba4f990a53b521204fbbb.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/images/3b9eafca8aab71a0761f38bfa35fdfbcb3a7fdff30eba4f990a53b521204fbbb.jpg)

![3dc88a36102414b5dbcd498dc4292f8e6f2e9d2145fd192862ce65752a60d087.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/images/3dc88a36102414b5dbcd498dc4292f8e6f2e9d2145fd192862ce65752a60d087.jpg)

### Tables

![1f5e77ae70e92187f539c7908492ca537d88de9d7a8a60f4786db6e3d76216ed.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/tables/1f5e77ae70e92187f539c7908492ca537d88de9d7a8a60f4786db6e3d76216ed.jpg)

![556e4de3e6f40c4cfb913c16b355134216067a75c926600d270ec4f0f2eea87c.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/tables/556e4de3e6f40c4cfb913c16b355134216067a75c926600d270ec4f0f2eea87c.jpg)

![8ef71fce122a2a13f076bdb0e022ec759761582354c93b707591fe2f038e3e66.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/tables/8ef71fce122a2a13f076bdb0e022ec759761582354c93b707591fe2f038e3e66.jpg)

![95f526e2c43f7c7b37ae88ccc2454f754985852b00a266d327da38a8de2bd5f1.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/tables/95f526e2c43f7c7b37ae88ccc2454f754985852b00a266d327da38a8de2bd5f1.jpg)

![e21c8526db17c23a724b7a0835b1fd089e20c7ab5bef9c7d6299849b85d07308.jpg](../icml_results/3792_Q-Delta_%20Beyond%20Key%E2%80%93Value%20Associative%20State%20Evolution/tables/e21c8526db17c23a724b7a0835b1fd089e20c7ab5bef9c7d6299849b85d07308.jpg)

<a id="paper-3753"></a>

## Class-Grouped Normalized Momentum and Faster Hyperparameter Exploration to Tackle Class Imbalance in Federated Learning


### Images

![038897a5218064dde1ea3385a5c6b523a6eb5bd716e2903bdeb3598117b31887.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/038897a5218064dde1ea3385a5c6b523a6eb5bd716e2903bdeb3598117b31887.jpg)

![07b3d08533ca76b6d7c92cf853a0cf6d0cd2c83fba6bad70fb98f3ff8de1771b.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/07b3d08533ca76b6d7c92cf853a0cf6d0cd2c83fba6bad70fb98f3ff8de1771b.jpg)

![07d2d9db99b371db2abbc9ef9d9f8c6c20bb8c79f730d0210ca762246771de09.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/07d2d9db99b371db2abbc9ef9d9f8c6c20bb8c79f730d0210ca762246771de09.jpg)

![1df0836ac15f2d9898370b0913e53de590d380c3dc49fa6376d4b1a36f86e483.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/1df0836ac15f2d9898370b0913e53de590d380c3dc49fa6376d4b1a36f86e483.jpg)

![1e47a72535063090458794a7f97dcac6530f1b9573f0c57c036b6b313d1e7d97.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/1e47a72535063090458794a7f97dcac6530f1b9573f0c57c036b6b313d1e7d97.jpg)

![3353fcb4a514ab38f876735b8dd6fdf976002be617091d388b1eb5e564c458d5.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/3353fcb4a514ab38f876735b8dd6fdf976002be617091d388b1eb5e564c458d5.jpg)

![7415bf48c2bc78abe0d26b1e15f0e87aad598dae6d10a40bf0e47bff9c08a884.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/7415bf48c2bc78abe0d26b1e15f0e87aad598dae6d10a40bf0e47bff9c08a884.jpg)

![79b82a18cbe3e7ce21fa34b4d055e5c2220aa8ffa3b924f054b608127910eae2.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/79b82a18cbe3e7ce21fa34b4d055e5c2220aa8ffa3b924f054b608127910eae2.jpg)

![96888f207d30a1e952af7b68493fd68fd86e96d5f4ef4083380f51000f4c2a76.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/96888f207d30a1e952af7b68493fd68fd86e96d5f4ef4083380f51000f4c2a76.jpg)

![a2c9730adc3b41447fa909523cb2b3bf5320f628f3f85c2f1e59273bd0035e3b.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/a2c9730adc3b41447fa909523cb2b3bf5320f628f3f85c2f1e59273bd0035e3b.jpg)

![ae1da233b6bd4c16082e329bcfa45ebc010318fc60871c12902bd6dd2dbb89b9.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/ae1da233b6bd4c16082e329bcfa45ebc010318fc60871c12902bd6dd2dbb89b9.jpg)

![d2cbb3d3f2c6c23d8400ad84f90757fd397100cff8bcb8b827c6082677f511ff.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/images/d2cbb3d3f2c6c23d8400ad84f90757fd397100cff8bcb8b827c6082677f511ff.jpg)

### Tables

![209dccac2203b7bec535c7d8084701603344faa7ca0630ad48ff8f4d86392a9b.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/209dccac2203b7bec535c7d8084701603344faa7ca0630ad48ff8f4d86392a9b.jpg)

![270ed6ffc520f5bb4f743e50b24d5a18e0fb964069b4f1ead5144e5376a3085d.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/270ed6ffc520f5bb4f743e50b24d5a18e0fb964069b4f1ead5144e5376a3085d.jpg)

![37e1d35d331951be2953398470fa1e8c60701de4bd8de1501f8263af74cdfa60.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/37e1d35d331951be2953398470fa1e8c60701de4bd8de1501f8263af74cdfa60.jpg)

![5430c81e4542cbab2ec0693371eddf180f83c97000b0ad323dab772e25200812.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/5430c81e4542cbab2ec0693371eddf180f83c97000b0ad323dab772e25200812.jpg)

![6564f07c09a0ad392b568774e0c48c3c7757da082160fadb10b207ce01a6a596.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/6564f07c09a0ad392b568774e0c48c3c7757da082160fadb10b207ce01a6a596.jpg)

![9aa121aff75ec43565a7fe9f432b242b7c79a7f94ae3eae52c77efb8b64409f1.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/9aa121aff75ec43565a7fe9f432b242b7c79a7f94ae3eae52c77efb8b64409f1.jpg)

![bb2f0ec68adedd545432e26a2713ce5765ea4aae9c6f91eb108611479ea301ef.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/bb2f0ec68adedd545432e26a2713ce5765ea4aae9c6f91eb108611479ea301ef.jpg)

![bc89459bcf01a54d6ab0e53a8beb2ae636b8930f4437bc2e3a03f9728a91498f.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/bc89459bcf01a54d6ab0e53a8beb2ae636b8930f4437bc2e3a03f9728a91498f.jpg)

![ce07426c26a96700624df5ac567fa5b0b50535acd2a6f0eb59d0ba3cd853e9ba.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/ce07426c26a96700624df5ac567fa5b0b50535acd2a6f0eb59d0ba3cd853e9ba.jpg)

![da135565e1a3a8e6701ad33c4a94353f7a9a046a1c6cdac31cf4f3ee2164586a.jpg](../icml_results/3793_Class-Grouped%20Normalized%20Momentum%20and%20Faster%20Hyperparameter%20Exploration%20to%20Tackle%20Class%20Imbalance%20in/tables/da135565e1a3a8e6701ad33c4a94353f7a9a046a1c6cdac31cf4f3ee2164586a.jpg)

<a id="paper-3754"></a>

## $\texttt{Multi}^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments


### Images

![0841dfcb43bd1b16ed8a770ee74ef5a4e41279ad69b3f844bfba7749be3b0df5.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/0841dfcb43bd1b16ed8a770ee74ef5a4e41279ad69b3f844bfba7749be3b0df5.jpg)

![14a5aae8b48d0d02d3f711e19fbdef0301b7a6b6a69c71d62159eb59a90208b7.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/14a5aae8b48d0d02d3f711e19fbdef0301b7a6b6a69c71d62159eb59a90208b7.jpg)

![152e6a2fa4fc0324384da4a20a8efe36bc94914c0093b913ab0167aa5cc6f594.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/152e6a2fa4fc0324384da4a20a8efe36bc94914c0093b913ab0167aa5cc6f594.jpg)

![24a0c52189b0b2038f4fba31bd196dc13e78921769c30e38d065025dfcfc3329.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/24a0c52189b0b2038f4fba31bd196dc13e78921769c30e38d065025dfcfc3329.jpg)

![26e13a7e5790eec924785f5c197d5002d373467dfa93379b364b80188d04c6c6.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/26e13a7e5790eec924785f5c197d5002d373467dfa93379b364b80188d04c6c6.jpg)

![3086d86a44aa275e9abf4d75c87bee990043c6547e42fff792596c6e5ea251f3.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/3086d86a44aa275e9abf4d75c87bee990043c6547e42fff792596c6e5ea251f3.jpg)

![360d535b23c6274db90248a08b8ac5fb1ad8120480f4a1fd37bf0eb4d124049e.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/360d535b23c6274db90248a08b8ac5fb1ad8120480f4a1fd37bf0eb4d124049e.jpg)

![3a0307b9b9230422416b88e3e19c58d1e48fe0c141a62f7a5ea79edfd553788b.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/3a0307b9b9230422416b88e3e19c58d1e48fe0c141a62f7a5ea79edfd553788b.jpg)

![4144a418d3949749668a12fe12464bddffa1e7fa7fe1af16b24c875fe877b0bf.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/4144a418d3949749668a12fe12464bddffa1e7fa7fe1af16b24c875fe877b0bf.jpg)

![4381a00956c0ce0c9590b4bc22f653f9d4748e5aaca7f8dd3f47c2b5f1654dd2.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/4381a00956c0ce0c9590b4bc22f653f9d4748e5aaca7f8dd3f47c2b5f1654dd2.jpg)

![4b30bf20be2121c22024731a68484ad9d0aa925c7a1bb22120ca3ea47552131d.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/4b30bf20be2121c22024731a68484ad9d0aa925c7a1bb22120ca3ea47552131d.jpg)

![4e7ef8054be07b7292f3e10859d5a232d7bb6dbffa64fa559b1bfa84fa465b68.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/4e7ef8054be07b7292f3e10859d5a232d7bb6dbffa64fa559b1bfa84fa465b68.jpg)

![4ff0c3f70124b26f92e7004bd4d2e4db007bf0743a23ea0426b77882c0bcbf54.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/4ff0c3f70124b26f92e7004bd4d2e4db007bf0743a23ea0426b77882c0bcbf54.jpg)

![56a8cc27dd123b88e89b5dcc2587b4c236ab193736f3d024fa8a0feab96ced60.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/56a8cc27dd123b88e89b5dcc2587b4c236ab193736f3d024fa8a0feab96ced60.jpg)

![6201b059b981c5195e240c9339a6c3a31989a055acc29a5f9c1b19dba9b5ec21.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/6201b059b981c5195e240c9339a6c3a31989a055acc29a5f9c1b19dba9b5ec21.jpg)

![6ebaca87f2ab900a59bb9bebb1fdb19bf285a6e90d952cd29e36a7af004d34be.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/6ebaca87f2ab900a59bb9bebb1fdb19bf285a6e90d952cd29e36a7af004d34be.jpg)

![7bb878071ff81f80c8adfde5a266e36ef7226542f519f471b73a4ec41f93bb62.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/7bb878071ff81f80c8adfde5a266e36ef7226542f519f471b73a4ec41f93bb62.jpg)

![7faa164017c391a55a71ad9f9242098b8edbb565fe8748b1951b44d56379ef59.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/7faa164017c391a55a71ad9f9242098b8edbb565fe8748b1951b44d56379ef59.jpg)

![87e388d77bef593cc72800691f43bd5a43ceae2a2827abce9b8d8450a70e5e17.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/87e388d77bef593cc72800691f43bd5a43ceae2a2827abce9b8d8450a70e5e17.jpg)

![9fc39000b28034c0719db4602011668b942341601039ec98a457d0f8e91bedcc.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/9fc39000b28034c0719db4602011668b942341601039ec98a457d0f8e91bedcc.jpg)

![bcd7f8609f6a11e1144a2c784d7a8be7ba8e604d79f710757c8364ad1c9da45f.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/bcd7f8609f6a11e1144a2c784d7a8be7ba8e604d79f710757c8364ad1c9da45f.jpg)

![c277c614322a6586f082234d406e45db07e74540f345b6be69d7746c6be39184.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/c277c614322a6586f082234d406e45db07e74540f345b6be69d7746c6be39184.jpg)

![e348b8cedba84e459abc382a96cfbf81899a5733492adea8bd0bbef4a5db2009.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/e348b8cedba84e459abc382a96cfbf81899a5733492adea8bd0bbef4a5db2009.jpg)

![e7a7a2055e012bab1b98a304760850e3c66c0d5ecaee6b6e648886bfd9f248ce.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/e7a7a2055e012bab1b98a304760850e3c66c0d5ecaee6b6e648886bfd9f248ce.jpg)

![ec8cf94aa5784440b90447a39b9457ede006542b814b2b7dcecd078977b9bbca.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/images/ec8cf94aa5784440b90447a39b9457ede006542b814b2b7dcecd078977b9bbca.jpg)

### Tables

![02fe1148a13c856be6e8a386395d62afdd5bab5adea7c43a2796de1572062803.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/02fe1148a13c856be6e8a386395d62afdd5bab5adea7c43a2796de1572062803.jpg)

![15d289d91d062477bd5eeeedc7e1fa18580006abb69925e82a3b507b49ff36f0.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/15d289d91d062477bd5eeeedc7e1fa18580006abb69925e82a3b507b49ff36f0.jpg)

![26c3f4e1c85861636e6437457a0e8b699fa5f24e1ea8452e6b2c63d1855f662c.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/26c3f4e1c85861636e6437457a0e8b699fa5f24e1ea8452e6b2c63d1855f662c.jpg)

![27ee382be4b641b49cb78b2821551d72143455862d8092b07da61d99934e41b0.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/27ee382be4b641b49cb78b2821551d72143455862d8092b07da61d99934e41b0.jpg)

![4244eea9f49b0b334bbc887b6f306e9dfc03c5a4b47f63afe31fbd16413a1ecd.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/4244eea9f49b0b334bbc887b6f306e9dfc03c5a4b47f63afe31fbd16413a1ecd.jpg)

![594b2c0d6cff01ec63717c8e15944866ddb07bcbef3d8aef205ead2753da99ef.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/594b2c0d6cff01ec63717c8e15944866ddb07bcbef3d8aef205ead2753da99ef.jpg)

![645e5d367215c38a445f4b565f871fa3c954d337570c3daa6c761060d3188c77.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/645e5d367215c38a445f4b565f871fa3c954d337570c3daa6c761060d3188c77.jpg)

![7f5b51b80efb873af6802c33007ace2b190bb742357c95646873834801ceb506.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/7f5b51b80efb873af6802c33007ace2b190bb742357c95646873834801ceb506.jpg)

![9ec5f5c38304d212443e6b7768ffcc88b7a4dbf2ed777ebe628aad94ed7a3c3a.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/9ec5f5c38304d212443e6b7768ffcc88b7a4dbf2ed777ebe628aad94ed7a3c3a.jpg)

![bc8c3c87a9e3e5f0912d48e0df5d2964fc8c4fc4c1fceac36a759bb759c19ce8.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/bc8c3c87a9e3e5f0912d48e0df5d2964fc8c4fc4c1fceac36a759bb759c19ce8.jpg)

![c1af3cabe1e195788c37ede47475c410b1ae42c2c9cc66c71611e0b75bc1d55e.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/c1af3cabe1e195788c37ede47475c410b1ae42c2c9cc66c71611e0b75bc1d55e.jpg)

![deabb64160a0c7d9b545ce8621d1554be5fb3fdb6eb61f01f1da4f037d3087dc.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/deabb64160a0c7d9b545ce8621d1554be5fb3fdb6eb61f01f1da4f037d3087dc.jpg)

![e4027cd49c405863e7ad2c297089ca301f227c66737d3927374f0150d0dec6ae.jpg](../icml_results/3794_%24_texttt%7BMulti%7D%5E2%24_%20Hierarchical%20Multi-Agent%20Decision-Making%20with%20LLM-Based%20Agents%20in%20Interactive%20En/tables/e4027cd49c405863e7ad2c297089ca301f227c66737d3927374f0150d0dec6ae.jpg)

<a id="paper-3755"></a>

## Euler–Poincaré Neural Dynamics: A Geometric-Mechanics Framework for Scientific Simulation


### Images

![09c99c28b97cdcfaa71dbb756e673e55499132e49ae0914be48dd3b37e793545.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/images/09c99c28b97cdcfaa71dbb756e673e55499132e49ae0914be48dd3b37e793545.jpg)

![255f4d67f25b40d09d9054142badca71f1bbf4dd27c07bc801852d5f1913311f.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/images/255f4d67f25b40d09d9054142badca71f1bbf4dd27c07bc801852d5f1913311f.jpg)

![509c273d543608fcd706b174c861733976ecdc722840b0b100bfc53df4f643e4.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/images/509c273d543608fcd706b174c861733976ecdc722840b0b100bfc53df4f643e4.jpg)

![6a7b87b466b5bf7e60c8247d7411092856dd769a174f7765e6238084194483c1.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/images/6a7b87b466b5bf7e60c8247d7411092856dd769a174f7765e6238084194483c1.jpg)

![b52ace2f30fec0043ac79dc9daf7105c110d1e26e01999caf7640beebebea92f.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/images/b52ace2f30fec0043ac79dc9daf7105c110d1e26e01999caf7640beebebea92f.jpg)

![d7fa9f0aa16f8deb6184d4958527c3643d288da6c7353c2067da15a6045f9755.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/images/d7fa9f0aa16f8deb6184d4958527c3643d288da6c7353c2067da15a6045f9755.jpg)

### Tables

![2246738c2c6c3b225cf8602b07a7875e19591464598dd2a51e863432fc734173.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/tables/2246738c2c6c3b225cf8602b07a7875e19591464598dd2a51e863432fc734173.jpg)

![43190ca292015d20d7a0e604bc41948b7dc43b993cd386b0c0eab68c496b9ff3.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/tables/43190ca292015d20d7a0e604bc41948b7dc43b993cd386b0c0eab68c496b9ff3.jpg)

![79bc3c319d4efac6b5228bdc28892436b431844499884c5dba5393a587dd5982.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/tables/79bc3c319d4efac6b5228bdc28892436b431844499884c5dba5393a587dd5982.jpg)

![9fecb1b08fc5ef0087e92eb7ae54ee302fd2b8709ac245214ed470ea675f0631.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/tables/9fecb1b08fc5ef0087e92eb7ae54ee302fd2b8709ac245214ed470ea675f0631.jpg)

![e0174962b605139b27989102a1b5defb2fb19595942abaa98d79a21013c467f0.jpg](../icml_results/3795_Euler%E2%80%93Poincar%C3%A9%20Neural%20Dynamics_%20A%20Geometric-Mechanics%20Framework%20for%20Scientific%20Simulation/tables/e0174962b605139b27989102a1b5defb2fb19595942abaa98d79a21013c467f0.jpg)

<a id="paper-3756"></a>

## Functional Adjoint Sampler: Scalable Sampling on Infinite Dimensional Spaces


### Images

![0d3f1bcdd3f8506160846d1d91f714c415810c7a5540eb98393962ee5e66bdc5.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/images/0d3f1bcdd3f8506160846d1d91f714c415810c7a5540eb98393962ee5e66bdc5.jpg)

![26bb6457304c5c525f0989d209eeae302d02a86b5286db7d2fa8a381466dae5b.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/images/26bb6457304c5c525f0989d209eeae302d02a86b5286db7d2fa8a381466dae5b.jpg)

![b08b41c1370b810078649dbaef98d9a7e93fdc9c5c037fee3d863c27393aa727.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/images/b08b41c1370b810078649dbaef98d9a7e93fdc9c5c037fee3d863c27393aa727.jpg)

![c08f720450eb80810825575e10da7101f848e638ea4e63fcda949beacc8268b9.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/images/c08f720450eb80810825575e10da7101f848e638ea4e63fcda949beacc8268b9.jpg)

![dff610d257df42e22db72b051ba0d838e6e699c68032d8f940a3689955cd7ba9.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/images/dff610d257df42e22db72b051ba0d838e6e699c68032d8f940a3689955cd7ba9.jpg)

![f8d658808edd209579a94adb13eb7776f9e8475dceb6922235fb4cbad68bba86.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/images/f8d658808edd209579a94adb13eb7776f9e8475dceb6922235fb4cbad68bba86.jpg)

### Tables

![0835332f04ea82687114a9cb0a9a83c55d2cc0d77da1b0979a4dd086aebdac54.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/tables/0835332f04ea82687114a9cb0a9a83c55d2cc0d77da1b0979a4dd086aebdac54.jpg)

![1f7f7f0655df080447612a2f09d8b39e04600ad28fbf63bfd4cf182e4286d858.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/tables/1f7f7f0655df080447612a2f09d8b39e04600ad28fbf63bfd4cf182e4286d858.jpg)

![b6be40ace5be22e0da84808a1ac11b7c7f5715dfacea709d7555fc5dce30d5bd.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/tables/b6be40ace5be22e0da84808a1ac11b7c7f5715dfacea709d7555fc5dce30d5bd.jpg)

![c6a8341b8c13cf9524831eb37db55864728e45b09cf046a48097115312fe40e9.jpg](../icml_results/3796_Functional%20Adjoint%20Sampler_%20Scalable%20Sampling%20on%20Infinite%20Dimensional%20Spaces/tables/c6a8341b8c13cf9524831eb37db55864728e45b09cf046a48097115312fe40e9.jpg)

<a id="paper-3757"></a>

## Saliency-Aware Model Merging


### Images

![16e03745f92dc709ffecb8bdacbdf81609ff1fe528e70a65578a58d57f361091.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/images/16e03745f92dc709ffecb8bdacbdf81609ff1fe528e70a65578a58d57f361091.jpg)

![3d6a2ae740f0f253b96119c5b61836bbfd9cc711febaed25ced91ac3cd6f40ab.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/images/3d6a2ae740f0f253b96119c5b61836bbfd9cc711febaed25ced91ac3cd6f40ab.jpg)

![e3aec9272b3f0f405e1faf42d731a7cd6c185c55738b886eeebe450568777e29.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/images/e3aec9272b3f0f405e1faf42d731a7cd6c185c55738b886eeebe450568777e29.jpg)

### Tables

![0d87de844a1a3e49ee54b5b9fba195a72a3b85d3e0f6ad5f8e7bd641444b148a.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/0d87de844a1a3e49ee54b5b9fba195a72a3b85d3e0f6ad5f8e7bd641444b148a.jpg)

![2db11d74a5dd39d926fcdeb5ed1352e8b46b719f21dcb09b1e9ec20d9c49f38c.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/2db11d74a5dd39d926fcdeb5ed1352e8b46b719f21dcb09b1e9ec20d9c49f38c.jpg)

![5064fa82636fd9ca20eedf27dd3d6df76a3c905031a3b5cf72f3f99b4d1de710.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/5064fa82636fd9ca20eedf27dd3d6df76a3c905031a3b5cf72f3f99b4d1de710.jpg)

![54ba65b45e2be827690055608aae2df95ee514555005c3af0d051e86a02cd749.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/54ba65b45e2be827690055608aae2df95ee514555005c3af0d051e86a02cd749.jpg)

![777def3cddce7a6e3a99537f8ba06efa6bfa73ff1e15b11ef82ab81e3e8eeb81.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/777def3cddce7a6e3a99537f8ba06efa6bfa73ff1e15b11ef82ab81e3e8eeb81.jpg)

![8c72bc175e7bdf4272cf85aa9786aeff87d8932ae16999731a81e009ee0e36fb.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/8c72bc175e7bdf4272cf85aa9786aeff87d8932ae16999731a81e009ee0e36fb.jpg)

![babef7caeb97e3dfade1d8bbcb3eb25423a9feba3f4f9cf5449308f8f6bc7881.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/babef7caeb97e3dfade1d8bbcb3eb25423a9feba3f4f9cf5449308f8f6bc7881.jpg)

![bc77f71bf0921b585fe8bfbeb7ec366b0a67ba32f608a3a005e9d66fff92c878.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/bc77f71bf0921b585fe8bfbeb7ec366b0a67ba32f608a3a005e9d66fff92c878.jpg)

![d5ec7b3b5a692d535d0b4882ca8d3109326634d85557984db461977d5b2dd602.jpg](../icml_results/3797_Saliency-Aware%20Model%20Merging/tables/d5ec7b3b5a692d535d0b4882ca8d3109326634d85557984db461977d5b2dd602.jpg)

<a id="paper-3758"></a>

## V-LynX: Token Interface Alignment for Video+X LLMs


### Images

![07db983ec64c5c98513835404e214e7514a674fffb151ab8ac2d945ce699895b.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/07db983ec64c5c98513835404e214e7514a674fffb151ab8ac2d945ce699895b.jpg)

![0b961a98c0d6dc116b630711301d17f700da972198d5a238d3af6045085de5ba.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/0b961a98c0d6dc116b630711301d17f700da972198d5a238d3af6045085de5ba.jpg)

![1ad1391689c88b20cdef528489f53b68c2353ce64cdb5fae44ffcc78264ff09e.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/1ad1391689c88b20cdef528489f53b68c2353ce64cdb5fae44ffcc78264ff09e.jpg)

![53aa1aea2bf9f7908345a0bc65cbc756e18786138da4db03f9c6c3ef5b348787.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/53aa1aea2bf9f7908345a0bc65cbc756e18786138da4db03f9c6c3ef5b348787.jpg)

![8252e15ec7eaaf35e1e85035fd1c8d85b0e9124064598d7917dd240ddebedc24.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/8252e15ec7eaaf35e1e85035fd1c8d85b0e9124064598d7917dd240ddebedc24.jpg)

![84c506fbef5c0e86874e7ddb446bc0cf0d8814f029b6f4826647f5a56f9c40e3.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/84c506fbef5c0e86874e7ddb446bc0cf0d8814f029b6f4826647f5a56f9c40e3.jpg)

![8574770c1c5241b008cbe801b8f790e6d2dfa5d6e7d39f3311db8cdf1b1fd797.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/8574770c1c5241b008cbe801b8f790e6d2dfa5d6e7d39f3311db8cdf1b1fd797.jpg)

![8df54f4fc3d596fda60b56e038b943227197ce2ca2703a5a57d61e0cf36e9b58.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/images/8df54f4fc3d596fda60b56e038b943227197ce2ca2703a5a57d61e0cf36e9b58.jpg)

### Tables

![06c8be56c80deb456a88b4c7e9e9aa78c4d4239e1306280db7cfa195c57df474.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/06c8be56c80deb456a88b4c7e9e9aa78c4d4239e1306280db7cfa195c57df474.jpg)

![0fd449231532f745b167810a47aaede86c6fface9f44e014c8561220dedc71ea.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/0fd449231532f745b167810a47aaede86c6fface9f44e014c8561220dedc71ea.jpg)

![4b16410d7edd3a5dd66f42012fa9dab985a5be996eff3a1af75976bcb87940c1.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/4b16410d7edd3a5dd66f42012fa9dab985a5be996eff3a1af75976bcb87940c1.jpg)

![70900d75c7d4a0e614ef5d5183e18d4b99a5b82fe32ac741ca1a5a8470a75ddc.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/70900d75c7d4a0e614ef5d5183e18d4b99a5b82fe32ac741ca1a5a8470a75ddc.jpg)

![a261f4944e5f30a3bb2b8d56cbb74ff1c952e54af73622e41d8ef721a1c311f5.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/a261f4944e5f30a3bb2b8d56cbb74ff1c952e54af73622e41d8ef721a1c311f5.jpg)

![a3fb1ddd7905d7dd664b1e3efd9f67c3e30f4c2c0dc4936026947244a3975368.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/a3fb1ddd7905d7dd664b1e3efd9f67c3e30f4c2c0dc4936026947244a3975368.jpg)

![a65e7740c529c3aaac305313cd05b536ce94aa4205c5c04b0bf00d21cf1f1eae.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/a65e7740c529c3aaac305313cd05b536ce94aa4205c5c04b0bf00d21cf1f1eae.jpg)

![da7aa959253547e8834d5f878040fe684ebc7a0c4d8fa4d678f155c0c3dbcf04.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/da7aa959253547e8834d5f878040fe684ebc7a0c4d8fa4d678f155c0c3dbcf04.jpg)

![ee2ac03cd7109f58255816ffa4ae54fcb1e9c7db7ac4a329a09530d2e6c4ef86.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/ee2ac03cd7109f58255816ffa4ae54fcb1e9c7db7ac4a329a09530d2e6c4ef86.jpg)

![f749942fe5832fc3f928b237bf7f5ff4589dbf1df476739682b934ff06809824.jpg](../icml_results/3798_V-LynX_%20Token%20Interface%20Alignment%20for%20Video%2BX%20LLMs/tables/f749942fe5832fc3f928b237bf7f5ff4589dbf1df476739682b934ff06809824.jpg)

<a id="paper-3759"></a>

## SleepMaMi: A Universal Sleep Foundation Model for Integrating Macro- and Micro-structures


### Images

![16b251da49e221421e26644b28e34012f913bfc4cd3ea11528cf922710bb42ce.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/16b251da49e221421e26644b28e34012f913bfc4cd3ea11528cf922710bb42ce.jpg)

![1dd5653828d6dcee6a76ac26c8a826966e0f81edb1f9ba0af9a68dcca3f8ce83.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/1dd5653828d6dcee6a76ac26c8a826966e0f81edb1f9ba0af9a68dcca3f8ce83.jpg)

![22e101895076fdff00fb4db2b7714c5d5b4df7d2942ba5d9b63a97e0f6ae4e0b.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/22e101895076fdff00fb4db2b7714c5d5b4df7d2942ba5d9b63a97e0f6ae4e0b.jpg)

![24a2587352f1ffd567d56301df2efc0e91ae19af320ec40862654c102193c850.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/24a2587352f1ffd567d56301df2efc0e91ae19af320ec40862654c102193c850.jpg)

![25fc0f6abc31114f4c501dfdd6649e6afc2fcb0149dff049f528fd23ffe96826.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/25fc0f6abc31114f4c501dfdd6649e6afc2fcb0149dff049f528fd23ffe96826.jpg)

![6605d942cb3104eda989e8db43c28ccb90402588a6f18d9995c82fdba52b74a3.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/6605d942cb3104eda989e8db43c28ccb90402588a6f18d9995c82fdba52b74a3.jpg)

![8e91a02167345f6ae17c25cf09eb8309d99ddf72b593332d2c6013cbf0b0c72d.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/8e91a02167345f6ae17c25cf09eb8309d99ddf72b593332d2c6013cbf0b0c72d.jpg)

![8ea549fb2700bff539fdd6be31a0682b4378a2897e1601a42d9631e0075c9f1a.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/8ea549fb2700bff539fdd6be31a0682b4378a2897e1601a42d9631e0075c9f1a.jpg)

![97b3c28c47b7b1730e5a96bc2613b6e32afb560edd0d399e7ad35c917ae24873.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/97b3c28c47b7b1730e5a96bc2613b6e32afb560edd0d399e7ad35c917ae24873.jpg)

![aa29cf959adbc404368f6df32048bb08d3ee1d93194e23bf2b63eb11a8f94adb.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/aa29cf959adbc404368f6df32048bb08d3ee1d93194e23bf2b63eb11a8f94adb.jpg)

![c0816a6928e9380a05ab1584f88684fd8dbe5dc1dff97314f0a346ee7004ff71.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/images/c0816a6928e9380a05ab1584f88684fd8dbe5dc1dff97314f0a346ee7004ff71.jpg)

### Tables

![00854769b7bdd0bf49e43c6241b653fc41e895e5b966acec96dbc688834c70a6.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/00854769b7bdd0bf49e43c6241b653fc41e895e5b966acec96dbc688834c70a6.jpg)

![00eb5706025e221197a41ce0a1fad4bd83445e35a6d821b149c983c7cf49e058.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/00eb5706025e221197a41ce0a1fad4bd83445e35a6d821b149c983c7cf49e058.jpg)

![129d9488599ded53a262a7450d8e6d4a972451513efdfd0c6879d0d71b2b4df6.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/129d9488599ded53a262a7450d8e6d4a972451513efdfd0c6879d0d71b2b4df6.jpg)

![20a98c2103cced6dd76c1012da7b75453339f9d47ba68a3263b8c4ce58ac2174.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/20a98c2103cced6dd76c1012da7b75453339f9d47ba68a3263b8c4ce58ac2174.jpg)

![28619c45cbea8ce370dabf83048a0d0f832e42cf4503138f8086ec83d5225715.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/28619c45cbea8ce370dabf83048a0d0f832e42cf4503138f8086ec83d5225715.jpg)

![41ab22a0f910481aeecc6d3a7db131944d7554a711a0be9e153a29698ccaed07.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/41ab22a0f910481aeecc6d3a7db131944d7554a711a0be9e153a29698ccaed07.jpg)

![49d05442753c902961db4c07b00095376c93f2c3ece5a6fc4be49d326cdb4be9.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/49d05442753c902961db4c07b00095376c93f2c3ece5a6fc4be49d326cdb4be9.jpg)

![55bc3554f8b6592a3f362a85c5b76b700c918cacc0bcdbdef749ab8b7799491f.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/55bc3554f8b6592a3f362a85c5b76b700c918cacc0bcdbdef749ab8b7799491f.jpg)

![58f821ef465a26e62d995d1d50099394a5c141a01564f966b641b2b049a8eba6.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/58f821ef465a26e62d995d1d50099394a5c141a01564f966b641b2b049a8eba6.jpg)

![5e9ffd87c04699322d0c7e31e94c1d79ad57b06fa1668959e43b905eda26bb64.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/5e9ffd87c04699322d0c7e31e94c1d79ad57b06fa1668959e43b905eda26bb64.jpg)

![606d26594c9ce835e5a02f89fa278302d413828e84b9a46f14a103865f6f5885.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/606d26594c9ce835e5a02f89fa278302d413828e84b9a46f14a103865f6f5885.jpg)

![6ac734c5cb7f5e47134733433dbe8ba4559046dfbd1663904962f8875934d1c1.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/6ac734c5cb7f5e47134733433dbe8ba4559046dfbd1663904962f8875934d1c1.jpg)

![aa00471ec8e2e7ad55c0832cdfce029a303d4c9c5bd22b7b19b06e13e23bfe0a.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/aa00471ec8e2e7ad55c0832cdfce029a303d4c9c5bd22b7b19b06e13e23bfe0a.jpg)

![ac839485f2eef484bb12dac0ae3a7bfb33b5d597390cc981ce4fcbb748308dea.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/ac839485f2eef484bb12dac0ae3a7bfb33b5d597390cc981ce4fcbb748308dea.jpg)

![b65a892dd221a1b9643f9cd0d2b62b408665f50f04545213b08c5a871af5165b.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/b65a892dd221a1b9643f9cd0d2b62b408665f50f04545213b08c5a871af5165b.jpg)

![b79a2e436eadcd56b2eef51a8c650d0a276b97196b8cbd083997c5bf059087a9.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/b79a2e436eadcd56b2eef51a8c650d0a276b97196b8cbd083997c5bf059087a9.jpg)

![d4590ff07439bf5ab89528d7b836bb29c27de0cceeaa851998e2d8aae7afcf3e.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/d4590ff07439bf5ab89528d7b836bb29c27de0cceeaa851998e2d8aae7afcf3e.jpg)

![f28366d363fcac997802f01a4afca9528232f8d5be9fc3d49f65b2f4c65a5423.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/f28366d363fcac997802f01a4afca9528232f8d5be9fc3d49f65b2f4c65a5423.jpg)

![f486913ce92f5e7be357cd674096425d2b2776b340f31ccba43fd29ea2905cd8.jpg](../icml_results/3799_SleepMaMi_%20A%20Universal%20Sleep%20Foundation%20Model%20for%20Integrating%20Macro-%20and%20Micro-structures/tables/f486913ce92f5e7be357cd674096425d2b2776b340f31ccba43fd29ea2905cd8.jpg)

<a id="paper-3760"></a>

## Constrained Multi-Objective Reinforcement Learning with Max-Min Criterion


### Tables

![1d2057aee37f2b0b9f18e9056b27055cc301e42b15aab5f716c86f276f804606.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/1d2057aee37f2b0b9f18e9056b27055cc301e42b15aab5f716c86f276f804606.jpg)

![451a839a112f72a8e14ddfa39bef4b58618f5143871aea62eaab510f669a0179.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/451a839a112f72a8e14ddfa39bef4b58618f5143871aea62eaab510f669a0179.jpg)

![68ed00c184f47dca2dc3f73cee9e6a7314c8ad06cc15dd704ad218a025b31efc.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/68ed00c184f47dca2dc3f73cee9e6a7314c8ad06cc15dd704ad218a025b31efc.jpg)

![7f0baeba95f1d72cdd6c91a1dc374a95d6886f1120c32540bf0f89030c5fd114.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/7f0baeba95f1d72cdd6c91a1dc374a95d6886f1120c32540bf0f89030c5fd114.jpg)

![926a171efa69208ca926dbe8ed59a7e10286fa8f2a778eefd1d67ba748d2face.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/926a171efa69208ca926dbe8ed59a7e10286fa8f2a778eefd1d67ba748d2face.jpg)

![939e7765457cba0f6fabacb7731df235170736274abb55e1beb0b933c283274a.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/939e7765457cba0f6fabacb7731df235170736274abb55e1beb0b933c283274a.jpg)

![a5bddd12dacf8bc03d7ebcdb15fccd7082c8ac8585d8450a597ef670d09e6a6a.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/a5bddd12dacf8bc03d7ebcdb15fccd7082c8ac8585d8450a597ef670d09e6a6a.jpg)

![b5a330a7a293434f8811c96159384ca4e8ef5f97266028db8b66cb150569ae76.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/b5a330a7a293434f8811c96159384ca4e8ef5f97266028db8b66cb150569ae76.jpg)

![c07eb61612b894e283af74afe99651f26cf0d6cf8ec27ab6c2d98703c6cb9677.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/c07eb61612b894e283af74afe99651f26cf0d6cf8ec27ab6c2d98703c6cb9677.jpg)

![cdf91c64f1d575bf678d6e9e0d17c10b95908cd7bd1b4e52406ec7ca9925456d.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/cdf91c64f1d575bf678d6e9e0d17c10b95908cd7bd1b4e52406ec7ca9925456d.jpg)

![ced86cc69b12ab6e5198d588c87dda9c87634db8e0839f524dccc369246557f7.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/ced86cc69b12ab6e5198d588c87dda9c87634db8e0839f524dccc369246557f7.jpg)

![f7c323c61eceb2a56539512fb1b9ffe16f6403bdd9315f0e8e5147b87e77b3e6.jpg](../icml_results/3800_Constrained%20Multi-Objective%20Reinforcement%20Learning%20with%20Max-Min%20Criterion/tables/f7c323c61eceb2a56539512fb1b9ffe16f6403bdd9315f0e8e5147b87e77b3e6.jpg)

<a id="paper-3761"></a>

## The Information Geometry of Softmax: Probing and Steering


### Images

![0b3caaadd62357ac400adde76ac38d52e3dd040d138edbf7835b08660dd5c4c8.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/0b3caaadd62357ac400adde76ac38d52e3dd040d138edbf7835b08660dd5c4c8.jpg)

![271a4c995ff208e324d91e63cd0af4e2f3bb7c3b742a74e4e16244fb399ec5e6.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/271a4c995ff208e324d91e63cd0af4e2f3bb7c3b742a74e4e16244fb399ec5e6.jpg)

![293eb6d2fb1314b231711301be5658507b816889ddffb6bd7e53afca4bd3f167.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/293eb6d2fb1314b231711301be5658507b816889ddffb6bd7e53afca4bd3f167.jpg)

![41ea2bed994df8cbc8e16eecb22ecc63c43f9cf38f80c714ad0762e31a771aec.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/41ea2bed994df8cbc8e16eecb22ecc63c43f9cf38f80c714ad0762e31a771aec.jpg)

![4bcf43e9b486196ea519d0d22975309644dfe489a58b38d9706b9054dff328b8.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/4bcf43e9b486196ea519d0d22975309644dfe489a58b38d9706b9054dff328b8.jpg)

![931e57efb2b0a0255595275c895ed3b021befe2cd877410ca97641254c8edd7d.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/931e57efb2b0a0255595275c895ed3b021befe2cd877410ca97641254c8edd7d.jpg)

![95cd00f14d0347a0b611a5715aec402d974abb336b4efaa97d1e47e1d5e378ca.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/95cd00f14d0347a0b611a5715aec402d974abb336b4efaa97d1e47e1d5e378ca.jpg)

![961f9ca014cf5b1ca6aa880839d5f124d26cbcc68d01901318e7bfc5fa0c7727.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/961f9ca014cf5b1ca6aa880839d5f124d26cbcc68d01901318e7bfc5fa0c7727.jpg)

![997c2dfc9e4a25ca951403d1802498c32c4e05eb0ad8cfcf11af9e8462748744.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/997c2dfc9e4a25ca951403d1802498c32c4e05eb0ad8cfcf11af9e8462748744.jpg)

![c88521a6eda380e7dd09233d081f034166c76467fed0c263ac23e2401d3460d8.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/c88521a6eda380e7dd09233d081f034166c76467fed0c263ac23e2401d3460d8.jpg)

![df1f019368d14f251331986dcf638ee9a01e2b6f9e94ea7159576d2323550f2e.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/df1f019368d14f251331986dcf638ee9a01e2b6f9e94ea7159576d2323550f2e.jpg)

![eec7581b4ee69b4e8f60f9da11b1225c3c16fd7509000b17daaf430f633a2d72.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/images/eec7581b4ee69b4e8f60f9da11b1225c3c16fd7509000b17daaf430f633a2d72.jpg)

### Tables

![6a1d528e2a84f27465b927a0c6e6e6b7718c0dabd7b8d9f6ec3f7dd48c3946c5.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/tables/6a1d528e2a84f27465b927a0c6e6e6b7718c0dabd7b8d9f6ec3f7dd48c3946c5.jpg)

![798de7d09ef2587d4e131bf2099dbb0ed6ce0820796f05f04fb9768526057376.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/tables/798de7d09ef2587d4e131bf2099dbb0ed6ce0820796f05f04fb9768526057376.jpg)

![806598cd6c23472e831230c3593ca9b6ad40c34ecec5a2f0fd5f5354af3d1c14.jpg](../icml_results/3801_The%20Information%20Geometry%20of%20Softmax_%20Probing%20and%20Steering/tables/806598cd6c23472e831230c3593ca9b6ad40c34ecec5a2f0fd5f5354af3d1c14.jpg)

<a id="paper-3762"></a>

## Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL

### Images

![232c0c46672a29e324872c11eb19892b7905a40c7d75260ecd1919d2d9f80ff0.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/232c0c46672a29e324872c11eb19892b7905a40c7d75260ecd1919d2d9f80ff0.jpg)

![35bed1cd8dc035bfb1b9efc280e37feff0cbd030197224dc7eb42116a0c0032a.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/35bed1cd8dc035bfb1b9efc280e37feff0cbd030197224dc7eb42116a0c0032a.jpg)

![53dd29cd8f7e0fdb50590a0d8dacbfe0fc02a7631e25db2d607f668f7804bebb.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/53dd29cd8f7e0fdb50590a0d8dacbfe0fc02a7631e25db2d607f668f7804bebb.jpg)

![67658730e775ab025cb325674cb4490a7ff667050dc1ad82da669d3623434e74.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/67658730e775ab025cb325674cb4490a7ff667050dc1ad82da669d3623434e74.jpg)

![7fe2eaea5bdbd3aedd98448a0bd323fe19cbc9be664cc5ba3bb9aa73be77a874.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/7fe2eaea5bdbd3aedd98448a0bd323fe19cbc9be664cc5ba3bb9aa73be77a874.jpg)

![8b75756cc855167e3c6cf4e0d40d3b5c876d3bf2f1cdfac331b7266c704cfd6f.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/8b75756cc855167e3c6cf4e0d40d3b5c876d3bf2f1cdfac331b7266c704cfd6f.jpg)

![9888d91af2977cafaaf554b2be310cce553ed520a99f24aaa3be6fbd0c392291.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/9888d91af2977cafaaf554b2be310cce553ed520a99f24aaa3be6fbd0c392291.jpg)

![b1ed6b7936318844ffc72836195e5a6e542bf071f5363d515afbfdb73cfc7b36.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/b1ed6b7936318844ffc72836195e5a6e542bf071f5363d515afbfdb73cfc7b36.jpg)

![d2cbe734b2444a2b21f81d5f0936c2e20fad53f8ce9d3188d8083ddb3d190939.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/d2cbe734b2444a2b21f81d5f0936c2e20fad53f8ce9d3188d8083ddb3d190939.jpg)

![e2bc3606399a01451e4e2dbcf7c2e7c4e808eeaee55e82d4f0ef0d7105cca756.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/e2bc3606399a01451e4e2dbcf7c2e7c4e808eeaee55e82d4f0ef0d7105cca756.jpg)

![e8e1f5eba91e6fecd197cf9b2f613c7bbca9010ca198891edbed06130b115120.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/images/e8e1f5eba91e6fecd197cf9b2f613c7bbca9010ca198891edbed06130b115120.jpg)

### Tables

![0be6d92d9ab5c273eb9894a17bf3784eb5697ce1d6502c69d122aaa6f15f8c68.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/0be6d92d9ab5c273eb9894a17bf3784eb5697ce1d6502c69d122aaa6f15f8c68.jpg)

![32ff13bac62d79cbd08f57e02474c2d004af066ad3df0eae843aeb77ea9ee4aa.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/32ff13bac62d79cbd08f57e02474c2d004af066ad3df0eae843aeb77ea9ee4aa.jpg)

![3c7d6b3d6853505cc3a0e7e76b93b9b2f4b575992f4fe1c3d4e6c2a11dfc1341.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/3c7d6b3d6853505cc3a0e7e76b93b9b2f4b575992f4fe1c3d4e6c2a11dfc1341.jpg)

![489c9982e8b0f96a6fc8338d4648f031177596ea125931d942941d43b5ddf768.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/489c9982e8b0f96a6fc8338d4648f031177596ea125931d942941d43b5ddf768.jpg)

![75afb0e95acd5a5fba09484f5cbde951064db4e5eed0e82c3a9a0b2b3535f88c.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/75afb0e95acd5a5fba09484f5cbde951064db4e5eed0e82c3a9a0b2b3535f88c.jpg)

![8d4e69edd3608336d52495390200f66fe5237e0bbba673960b02897b2279e84a.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/8d4e69edd3608336d52495390200f66fe5237e0bbba673960b02897b2279e84a.jpg)

![c8faa0a096e28d36cd1ea5aa6fe76b9b12db0fb20ac632e71d6d61eec30c5efb.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/c8faa0a096e28d36cd1ea5aa6fe76b9b12db0fb20ac632e71d6d61eec30c5efb.jpg)

![ef1a141f610ce69f1f42e2d379e061c195638e0a6a3ac64efecf08873e901bab.jpg](../icml_results/3802_Learning%20Generalizable%20Skill%20Policy%20with%20Data-Efficient%20Unsupervised%20RL/tables/ef1a141f610ce69f1f42e2d379e061c195638e0a6a3ac64efecf08873e901bab.jpg)
