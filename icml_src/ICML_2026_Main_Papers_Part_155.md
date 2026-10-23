# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 155 of 197

## 目录 (Table of Contents)

1. [The Mechanistic Emergence of Symbol Grounding in Language Models](#paper-5083)
2. [MulFCoder: Framework-conditioned Multi-agent for MLLM-based Multi-framework Front-end Code Generation](#paper-5084)
3. [Are Large Reasoning Models Interruptible?](#paper-5085)
4. [RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience](#paper-5086)
5. [AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines](#paper-5087)
6. [Reasoning Cache: Continual Improvement Over Long Horizons via Short-Horizon RL](#paper-5088)
7. [REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models](#paper-5089)
8. [$f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses](#paper-5090)
9. [Genome-Factory: A Library for Tuning, Deploying, and Interpreting Genomic Foundation Models](#paper-5091)
10. [Attention Sinks in Diffusion Transformers: A Causal Analysis](#paper-5092)
11. [PortraitRL: Reinforcement Learning for Portrait Pose Transfer with Multi-Objective Reward](#paper-5093)
12. [Beyond Majority Voting: Self-Reflective Test-Time Reinforcement Learning for LLM Reasoning](#paper-5094)
13. [From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG](#paper-5095)
14. [Score Correction for Generative Models with Probabilistic Constraints](#paper-5096)
15. [Reasoning Compartmentalization: Bridging the Concretization Gap via Abstraction-based Routing](#paper-5097)
16. [ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation](#paper-5098)
17. [Letting Trajectories Spread: Quality-Preserving Control for Diverse Flow Matching](#paper-5099)
18. [From Interactions to Principles: Experience-Driven Self-Distillation for Evolving LLM Agents](#paper-5100)
19. [Language Model Networks: Supervision-Efficient Learning through Dense Communication](#paper-5101)
20. [Time-Series Decomposition as a Standalone Task: A Mechanism-Driven Diagnostic Benchmark](#paper-5102)
21. [Neural–Evolutionary Symbolic Regression with Global Constraints: Constraint-Aware Decoding and Reward Shaping](#paper-5103)
22. [One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery](#paper-5104)
23. [When Data Is Scarce: Scaling Sparse Language Models with Repeated Training](#paper-5105)
24. [CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning](#paper-5106)
25. [HiST: A Hierarchical Sparse Transformer for Cross-Modal Spatial Transcriptomics Modeling](#paper-5107)
26. [PnP-Corrector: A Universal Correction Framework for Coupled Spatiotemporal Forecasting](#paper-5108)
27. [From Interaction Trajectories to Prompt Rules: Credit Assignment for Multi-Agent Prompt Optimization](#paper-5109)
28. [Proteo-R1: Reasoning Foundation Models for De Novo Protein Design](#paper-5110)
29. [Trajectory-Aware Certified Decentralized Unlearning via SGD Stability](#paper-5111)
30. [Why Self-Distillation Helps and Hurts: Denoising vs. Signal Forgetting](#paper-5112)
31. [Speedup Patch: Learning a Plug-and-Play Policy to Accelerate Embodied Manipulation](#paper-5113)
32. [Chain-of-Glimpse: Search-Guided Progressive Object-Grounded Reasoning for Video Understanding](#paper-5114)
33. [Furina: Fragmented Uncertainty-Driven Refusal Instability Attack](#paper-5115)

---


<a id="paper-5083"></a>

## The Mechanistic Emergence of Symbol Grounding in Language Models

### Images

![1ba7e3afdad77d3ab06e15e5189b16c2faa95bf97f923d344efbec3d620ba817.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/1ba7e3afdad77d3ab06e15e5189b16c2faa95bf97f923d344efbec3d620ba817.jpg)

![1e47ec9cd05d2f908c6a2ca234e1f2bfcba1e1cf8128c1cf17f08dca1e6cf0f7.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/1e47ec9cd05d2f908c6a2ca234e1f2bfcba1e1cf8128c1cf17f08dca1e6cf0f7.jpg)

![1e850a2a1edfc07559aa30f47f56866c78fcc095adc3eadf1eddeb261783db91.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/1e850a2a1edfc07559aa30f47f56866c78fcc095adc3eadf1eddeb261783db91.jpg)

![46637e3bd21ff14f04e6d1a545a508b6d771bc2b2d838c69c176009c3174ec8f.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/46637e3bd21ff14f04e6d1a545a508b6d771bc2b2d838c69c176009c3174ec8f.jpg)

![622d617993a8c197d839804da9d5d87e8fbfefc67285aa68527ebc9363f8e072.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/622d617993a8c197d839804da9d5d87e8fbfefc67285aa68527ebc9363f8e072.jpg)

![93df79d4d02b000fdae1ab61e3fc7bf95d7bac878fdd918df06e4f9218f97cf9.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/93df79d4d02b000fdae1ab61e3fc7bf95d7bac878fdd918df06e4f9218f97cf9.jpg)

![9c0832577e55c5dd13eb7d5f1d826e6633b8b5db595d8c6accdee269f0324b06.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/9c0832577e55c5dd13eb7d5f1d826e6633b8b5db595d8c6accdee269f0324b06.jpg)

![ca53372b5251e7bbc24345d9a371f1e5bff4d6bbc01462b5d657a6e0433f1d7e.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/ca53372b5251e7bbc24345d9a371f1e5bff4d6bbc01462b5d657a6e0433f1d7e.jpg)

![dc317663e5cd9e9bf2f5d132325a21298f57e2e7db466e627b1520a04e39ee7b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/dc317663e5cd9e9bf2f5d132325a21298f57e2e7db466e627b1520a04e39ee7b.jpg)

![f71044d0fa600fe6ce2bef91ff4571eba4095e44e0b9a7e22c7c410bc56e9a94.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/f71044d0fa600fe6ce2bef91ff4571eba4095e44e0b9a7e22c7c410bc56e9a94.jpg)

![fb83694a5905fe0d4362db07ccec6ae4caf23f45c6b72a1043d599a3fa7c7f74.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/images/fb83694a5905fe0d4362db07ccec6ae4caf23f45c6b72a1043d599a3fa7c7f74.jpg)

### Tables

![0ef5034e025f44ec92b0aa818c5419c83dcd2f15743cde216ceb6af0a6e18732.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/0ef5034e025f44ec92b0aa818c5419c83dcd2f15743cde216ceb6af0a6e18732.jpg)

![112570d5e5f0dc26f4f40f9c2be4118c8e949121e8adf35acd861bcc0427a01b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/112570d5e5f0dc26f4f40f9c2be4118c8e949121e8adf35acd861bcc0427a01b.jpg)

![12a755395589903037d38e049e5f4afa8206efb10ed5bdda168bdb9da1fcd7a0.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/12a755395589903037d38e049e5f4afa8206efb10ed5bdda168bdb9da1fcd7a0.jpg)

![1842af13b4483316610d41cedc443f9c63393b8f7b1d58fa53df3cecf729e8b0.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1842af13b4483316610d41cedc443f9c63393b8f7b1d58fa53df3cecf729e8b0.jpg)

![1daf72c66e2da4ee6b5d867f0c859a05a224bf78954f95aa7e0a4c91510d4af7.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1daf72c66e2da4ee6b5d867f0c859a05a224bf78954f95aa7e0a4c91510d4af7.jpg)

![1db5c06c93d6f8b73fc5126c6a45885d90293499729f62473e387b598297f1b3.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1db5c06c93d6f8b73fc5126c6a45885d90293499729f62473e387b598297f1b3.jpg)

![1e4224db95cfe16e1b066febdbbe5cb9a3c508d3b08a6fc484868eb265a8f82e.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/1e4224db95cfe16e1b066febdbbe5cb9a3c508d3b08a6fc484868eb265a8f82e.jpg)

![2247ab4469d30bab4fa317ed81bdf21a24f08ca063d97c3f2f40191e929e984b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/2247ab4469d30bab4fa317ed81bdf21a24f08ca063d97c3f2f40191e929e984b.jpg)

![3189b826b1b17418431fca87d3a38381815584318c669f9a9a289556958beadf.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3189b826b1b17418431fca87d3a38381815584318c669f9a9a289556958beadf.jpg)

![35319cdf66b60d3ec3d5b897e0d9e511f365e042e120b9c5e476178fac3e69e9.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/35319cdf66b60d3ec3d5b897e0d9e511f365e042e120b9c5e476178fac3e69e9.jpg)

![3607c48743215ae51ae1f110887b52d628bc1386eb0c9fc75191c4d656d00029.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3607c48743215ae51ae1f110887b52d628bc1386eb0c9fc75191c4d656d00029.jpg)

![3d0805fc4a167cd7114098c257a0e73eed2e1e3f47d7f9847d68200492c8f555.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3d0805fc4a167cd7114098c257a0e73eed2e1e3f47d7f9847d68200492c8f555.jpg)

![3d93b56fa808924ad157790c8c2b1e294e4fe5824ad02d782e10bea4089cbe4a.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/3d93b56fa808924ad157790c8c2b1e294e4fe5824ad02d782e10bea4089cbe4a.jpg)

![40bed2a50b2b111e2096a72fe670137405f49548b9162b30e9871e84c1d5bb76.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/40bed2a50b2b111e2096a72fe670137405f49548b9162b30e9871e84c1d5bb76.jpg)

![4cd3c71297254711ccfcc759fe99030dbd5298144e5a736ac9aab0f34808b070.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/4cd3c71297254711ccfcc759fe99030dbd5298144e5a736ac9aab0f34808b070.jpg)

![527ad707fcf8f6aeaaa36dc226d682e3c97f234be69e41e0da67163ed27020c9.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/527ad707fcf8f6aeaaa36dc226d682e3c97f234be69e41e0da67163ed27020c9.jpg)

![5535caf649563b1d87cd282707a62f1e05f9b201c5ca7f8e08b563070e1f7327.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/5535caf649563b1d87cd282707a62f1e05f9b201c5ca7f8e08b563070e1f7327.jpg)

![5ec4853be9678ebd6435fa9e165b980a5752e9525ee95c12bb11ca63a94740ea.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/5ec4853be9678ebd6435fa9e165b980a5752e9525ee95c12bb11ca63a94740ea.jpg)

![6457235ffc417eb00f0e5ac87848685eb8185a9967cc321327ee1431c49c6fbd.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/6457235ffc417eb00f0e5ac87848685eb8185a9967cc321327ee1431c49c6fbd.jpg)

![68003786df293f66a4d222bf8ebb74f2c2f042884074c0f2cf33b3c527b6d276.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/68003786df293f66a4d222bf8ebb74f2c2f042884074c0f2cf33b3c527b6d276.jpg)

![6917dfaafcc602245c359719b73af2372dce0512d37a52f69109d77732db7281.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/6917dfaafcc602245c359719b73af2372dce0512d37a52f69109d77732db7281.jpg)

![712d34f8c51305a6a8a9c6d8f29bbbd584d5de5b3a5d050c1bb48ec3d257397b.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/712d34f8c51305a6a8a9c6d8f29bbbd584d5de5b3a5d050c1bb48ec3d257397b.jpg)

![7f35523cc6de5c11bd7ff86b6a4bbe5db477a94cfaca72d3a07e170969f0c556.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/7f35523cc6de5c11bd7ff86b6a4bbe5db477a94cfaca72d3a07e170969f0c556.jpg)

![80f9fe2c2920affb266e31e78a1d764f073ef3b3b1e26ca1b6ef22a6216c2394.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/80f9fe2c2920affb266e31e78a1d764f073ef3b3b1e26ca1b6ef22a6216c2394.jpg)

![9aa1ef29de4521386aa1c3a7a4b702c9db64e161b985bc7d7bc6cba4880969f0.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/9aa1ef29de4521386aa1c3a7a4b702c9db64e161b985bc7d7bc6cba4880969f0.jpg)

![a5ac836ff87c5e582feafe0eab16567b6e468a95b33c4de1ca5744edf0cf3461.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/a5ac836ff87c5e582feafe0eab16567b6e468a95b33c4de1ca5744edf0cf3461.jpg)

![b876e3c17102c8cfcdfbcb36e6db8080210b39a16113c9a00e5a6fe0c4b8faef.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/b876e3c17102c8cfcdfbcb36e6db8080210b39a16113c9a00e5a6fe0c4b8faef.jpg)

![d085c86a0114d456e3ae23aab28bf442551e4f220d3c2e37e218a55addef3906.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/d085c86a0114d456e3ae23aab28bf442551e4f220d3c2e37e218a55addef3906.jpg)

![d2940905ed1bb5ee590605673af16e980651a85554aae0336dadb8ad3ba4c12d.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/d2940905ed1bb5ee590605673af16e980651a85554aae0336dadb8ad3ba4c12d.jpg)

![dc90c42308cf808c96264ac64d489215b4e883b988d1c6d7e08e31d4a2c492bd.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/dc90c42308cf808c96264ac64d489215b4e883b988d1c6d7e08e31d4a2c492bd.jpg)

![de21235675fb37bf26ec2196ee9ea2b771d08c966948b7c3ec8e6b3735543273.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/de21235675fb37bf26ec2196ee9ea2b771d08c966948b7c3ec8e6b3735543273.jpg)

![f2ab140710684ade97b0ebd2162f4abe87c85cce57721e9255a1824958323f9f.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/f2ab140710684ade97b0ebd2162f4abe87c85cce57721e9255a1824958323f9f.jpg)

![f3169daf095bf0f57428b1a8136230d384d2f394cd24b8e23ea8608ec255f3b8.jpg](../icml_results/5130_Can%20Agents%20Generalize%20to%20the%20Open%20World_%20Unveiling%20the%20Fragility%20of%20Static%20Training%20in%20Tool%20Use/tables/f3169daf095bf0f57428b1a8136230d384d2f394cd24b8e23ea8608ec255f3b8.jpg)

## The Mechanistic Emergence of Symbol Grounding in Language Models


### Images

![18c4b2e97cfd249d5e6331458bb82e6d493cd7dfd47de118aff77590a9a601f2.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/18c4b2e97cfd249d5e6331458bb82e6d493cd7dfd47de118aff77590a9a601f2.jpg)

![28e632505511a2fe0041714ee9008e947788f7f3be8b2cd1d4b076e2bdf465d9.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/28e632505511a2fe0041714ee9008e947788f7f3be8b2cd1d4b076e2bdf465d9.jpg)

![2b442cf89c487706e184afffeb91b2a81f2866cd83ad8ffbf50ce6f9d44c1eed.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/2b442cf89c487706e184afffeb91b2a81f2866cd83ad8ffbf50ce6f9d44c1eed.jpg)

![2b6ac0c7ecf5be8287b08c0baf7c83f91386c79b7c13939a8b48039a5c7b4d63.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/2b6ac0c7ecf5be8287b08c0baf7c83f91386c79b7c13939a8b48039a5c7b4d63.jpg)

![41b07511f103e2f2182b67007e697da575d429251e3a069ce97700e003f44aef.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/41b07511f103e2f2182b67007e697da575d429251e3a069ce97700e003f44aef.jpg)

![5475d27e04ee221d86c75eb18ce9c15f3dbabd6107651d1afb05f51d9e9d61ed.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/5475d27e04ee221d86c75eb18ce9c15f3dbabd6107651d1afb05f51d9e9d61ed.jpg)

![95c5f23d6205a5d72fec736ab33e6bc1abf2b3e7eda66c2d7243d6b393ff85ff.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/95c5f23d6205a5d72fec736ab33e6bc1abf2b3e7eda66c2d7243d6b393ff85ff.jpg)

![a1141ee0f32c98e063b2837f5b297f18a8f0de429643e86ec0ff1276a467ccfc.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/a1141ee0f32c98e063b2837f5b297f18a8f0de429643e86ec0ff1276a467ccfc.jpg)

![a2d1b7713e3ec5817cbf937b2ad69da78d3838aef4826590a3c0bafb98c44fb1.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/a2d1b7713e3ec5817cbf937b2ad69da78d3838aef4826590a3c0bafb98c44fb1.jpg)

![abcea2d8bb8abf9b8ff4513ddb5746578cd5e66ab6583030249296044a40cc68.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/abcea2d8bb8abf9b8ff4513ddb5746578cd5e66ab6583030249296044a40cc68.jpg)

![b3a9d1ae55d7c6c1285b93dd2c3e988190ae3b3f6a5bb48ff5062aef292d5f37.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/b3a9d1ae55d7c6c1285b93dd2c3e988190ae3b3f6a5bb48ff5062aef292d5f37.jpg)

![cae902f142b1fd9854b2a12a5566a2586694d2cc6a6c3ca89dcf40cb9a0b896d.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/cae902f142b1fd9854b2a12a5566a2586694d2cc6a6c3ca89dcf40cb9a0b896d.jpg)

![ed55b25fc63687356981e3f764e8f3b9ee1e23caa9fe7053ab7717d2369b0b7d.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/images/ed55b25fc63687356981e3f764e8f3b9ee1e23caa9fe7053ab7717d2369b0b7d.jpg)

### Tables

![3977ec8c458d6f4bb001f0bd942f4dd21e18ecbf92e4eb81d8389e2d1f8edded.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/tables/3977ec8c458d6f4bb001f0bd942f4dd21e18ecbf92e4eb81d8389e2d1f8edded.jpg)

![49f17017e9087f381298d6275ef86de816e17558b3cd7f1769c8b0c3d40626d6.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/tables/49f17017e9087f381298d6275ef86de816e17558b3cd7f1769c8b0c3d40626d6.jpg)

![4d13ddf92b24308084f68ff0fec36d67c8bae9d698b5f91a3dcc73112c029118.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/tables/4d13ddf92b24308084f68ff0fec36d67c8bae9d698b5f91a3dcc73112c029118.jpg)

![d67ccc706a6e688de418a9d0b42e2a7915025f644d4264e42ca3f5fb195575c1.jpg](../icml_results/5131_The%20Mechanistic%20Emergence%20of%20Symbol%20Grounding%20in%20Language%20Models/tables/d67ccc706a6e688de418a9d0b42e2a7915025f644d4264e42ca3f5fb195575c1.jpg)

<a id="paper-5084"></a>

## MulFCoder: Framework-conditioned Multi-agent for MLLM-based Multi-framework Front-end Code Generation


### Images

![366d79208a557f9a6d21c7766ca19ff906368799eb27e9d9b21b295a528e31cb.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/images/366d79208a557f9a6d21c7766ca19ff906368799eb27e9d9b21b295a528e31cb.jpg)

![405d71ae07b1223e9aca498aa9b52f25221a8bc30eaf3ef88cf995b2a1df05ca.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/images/405d71ae07b1223e9aca498aa9b52f25221a8bc30eaf3ef88cf995b2a1df05ca.jpg)

![7abdf6718da258d2e27927692841a57d872b84c16c4bdf154ada325d4c345b36.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/images/7abdf6718da258d2e27927692841a57d872b84c16c4bdf154ada325d4c345b36.jpg)

![89e59f0b922a616f01f70e106308a0a0bac7cac03fb2ecd79257f0c9c187acbd.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/images/89e59f0b922a616f01f70e106308a0a0bac7cac03fb2ecd79257f0c9c187acbd.jpg)

![a02f2cada58dc12385f8617aad931ee2190ce30ba8d98ae768fba185c0555b7d.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/images/a02f2cada58dc12385f8617aad931ee2190ce30ba8d98ae768fba185c0555b7d.jpg)

### Tables

![0f29e410a1621ae2beff65e13b35c5ebea44b0f8e47f9f31f58951d391d34538.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/tables/0f29e410a1621ae2beff65e13b35c5ebea44b0f8e47f9f31f58951d391d34538.jpg)

![5fd80b35bdf27899184857b7fc9a2a8ce7594ea0f12731730505184ec1ead279.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/tables/5fd80b35bdf27899184857b7fc9a2a8ce7594ea0f12731730505184ec1ead279.jpg)

![7621afe9b343cf6359c8430653a2b4df3f1e4985764c4e5d63ae875779476ad9.jpg](../icml_results/5132_MulFCoder_%20Framework-conditioned%20Multi-agent%20for%20MLLM-based%20Multi-framework%20Front-end%20Code%20Generatio/tables/7621afe9b343cf6359c8430653a2b4df3f1e4985764c4e5d63ae875779476ad9.jpg)

<a id="paper-5085"></a>

## Are Large Reasoning Models Interruptible?


### Images

![01096a06720ae938c0fdc0b8edee60256c9bb612d6f48e256dca9ade6bf54e61.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/01096a06720ae938c0fdc0b8edee60256c9bb612d6f48e256dca9ade6bf54e61.jpg)

![17aad589da92c9c0daa44ac5544af3c81b987e02b2afaf6dc11f0c5ef829d035.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/17aad589da92c9c0daa44ac5544af3c81b987e02b2afaf6dc11f0c5ef829d035.jpg)

![1b83bfee9db80815df216d8084ec206d8250e69acb68d906b924625b5a2364f9.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/1b83bfee9db80815df216d8084ec206d8250e69acb68d906b924625b5a2364f9.jpg)

![218887defb9be67bb795bfbc8679ddbbddfea1e18001201f69861be83608b55e.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/218887defb9be67bb795bfbc8679ddbbddfea1e18001201f69861be83608b55e.jpg)

![218e47ace0ed3abd9bee40247e15444efa3d3234e4cca9474295e9f88339b256.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/218e47ace0ed3abd9bee40247e15444efa3d3234e4cca9474295e9f88339b256.jpg)

![2c5514fca6772605c0b7e30f58239b8e0e4f8af06d532d89be4c15f02d380503.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/2c5514fca6772605c0b7e30f58239b8e0e4f8af06d532d89be4c15f02d380503.jpg)

![2d64725fead5ea7c465c4e88178f124e8bb1f2d5c4c12bf97b06eba46202d3cb.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/2d64725fead5ea7c465c4e88178f124e8bb1f2d5c4c12bf97b06eba46202d3cb.jpg)

![32957b6308629c37283de369ad0230214654d8074970e81940894af113f40787.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/32957b6308629c37283de369ad0230214654d8074970e81940894af113f40787.jpg)

![3e8cf77a5c8a0cccb4f6072b755ba06ba5b1917d8a2d665753dd5fa1aa326dab.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/3e8cf77a5c8a0cccb4f6072b755ba06ba5b1917d8a2d665753dd5fa1aa326dab.jpg)

![3f2a37f7265a37d47fd7b7420a4a849f5f35594c78adb96bc56be0b473213961.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/3f2a37f7265a37d47fd7b7420a4a849f5f35594c78adb96bc56be0b473213961.jpg)

![4cf5c26b4f33749b4b6d46f00a141d6297d36e6e49a1cd09a92852eddc4a1058.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/4cf5c26b4f33749b4b6d46f00a141d6297d36e6e49a1cd09a92852eddc4a1058.jpg)

![4f1f94ca244906a67dd9c4dbe124054e0110f3a9e5961d1ce5f47c9239e389a7.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/4f1f94ca244906a67dd9c4dbe124054e0110f3a9e5961d1ce5f47c9239e389a7.jpg)

![526dac433db51650628cd5da9c16286be19ef679608bb976a9904cdecb0513b6.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/526dac433db51650628cd5da9c16286be19ef679608bb976a9904cdecb0513b6.jpg)

![5617ee6fa9128bed6eaf26f6810bba4e8b8e34bc610db958dff1faf3c18babf4.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/5617ee6fa9128bed6eaf26f6810bba4e8b8e34bc610db958dff1faf3c18babf4.jpg)

![57dd605b0fbb260ee88256364d0d615f890dfacc6e361a33d23d8340540656d0.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/57dd605b0fbb260ee88256364d0d615f890dfacc6e361a33d23d8340540656d0.jpg)

![599c91410c82ea45d4d2c39fce13d34d7ae9fa51bb4bf33f72bb7871daf48af5.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/599c91410c82ea45d4d2c39fce13d34d7ae9fa51bb4bf33f72bb7871daf48af5.jpg)

![6311cb0481ee09e06b46e211008fc89fb8f16410065a640e19b73c5e4622c1af.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/6311cb0481ee09e06b46e211008fc89fb8f16410065a640e19b73c5e4622c1af.jpg)

![66a19e651d49367f1d08a563821731cc57f487b0e1c88aee0d3a882c3e38c2ea.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/66a19e651d49367f1d08a563821731cc57f487b0e1c88aee0d3a882c3e38c2ea.jpg)

![6a8af618bffc91afa63d0a2c1ba63e875c7de3b4dbc09a91ec11c6c01e11d55b.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/6a8af618bffc91afa63d0a2c1ba63e875c7de3b4dbc09a91ec11c6c01e11d55b.jpg)

![6ecb96c242c60bb63004223731cc9159552782e6162f4f3b047b9b50b89b7838.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/6ecb96c242c60bb63004223731cc9159552782e6162f4f3b047b9b50b89b7838.jpg)

![71fe23e33601ac93453a641c3bce176e46da8b8294309e1dbd274e36de45e976.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/71fe23e33601ac93453a641c3bce176e46da8b8294309e1dbd274e36de45e976.jpg)

![914c24e6962e954349a958cc375e1c04aebb73d122c4be11b99bd6181ce67e39.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/914c24e6962e954349a958cc375e1c04aebb73d122c4be11b99bd6181ce67e39.jpg)

![978fd2a2c24257a1a650215bdd0ccd3c1dfdd3acf7f61bf0610b66fbd46a580b.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/978fd2a2c24257a1a650215bdd0ccd3c1dfdd3acf7f61bf0610b66fbd46a580b.jpg)

![97eed1d291df94a4e1e9f37acc74ea08bbe2ab5c4fd9b7cd11e558266798f409.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/97eed1d291df94a4e1e9f37acc74ea08bbe2ab5c4fd9b7cd11e558266798f409.jpg)

![9f27d290312c0d2c868cd53f2987126850d2405fd03f7f4fe612d688ed3278ed.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/9f27d290312c0d2c868cd53f2987126850d2405fd03f7f4fe612d688ed3278ed.jpg)

![a1e945641b88e77059fe653f487280d230b56e5cf43455597212274895025f2a.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/a1e945641b88e77059fe653f487280d230b56e5cf43455597212274895025f2a.jpg)

![a584278b8cfdcaaa6a3b5b4981abb0560c0262daf20ae1b96b2e2295a7678812.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/a584278b8cfdcaaa6a3b5b4981abb0560c0262daf20ae1b96b2e2295a7678812.jpg)

![aaaa843fb1deb89cbdb5863059bd2dad5e446b45f66107ce802fa542e0fb9a7a.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/aaaa843fb1deb89cbdb5863059bd2dad5e446b45f66107ce802fa542e0fb9a7a.jpg)

![afa30562c5bb9d12bce0d1db3252cf5171be5d0fb6b0653dc8341bf55041c4d3.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/afa30562c5bb9d12bce0d1db3252cf5171be5d0fb6b0653dc8341bf55041c4d3.jpg)

![b73298fdb1ff9baf8a6204688fafa934cc93c83c6123b1acfa757e5210b70e4f.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/b73298fdb1ff9baf8a6204688fafa934cc93c83c6123b1acfa757e5210b70e4f.jpg)

![c340493b5beffcf9d6ef5004cd7f044a20349faef29ee7b2846facc6d385d932.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/c340493b5beffcf9d6ef5004cd7f044a20349faef29ee7b2846facc6d385d932.jpg)

![c5cc2306fab221fa2637a7e55967411d2033d6fb875f4cf0c7a7c04a3864f4bd.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/c5cc2306fab221fa2637a7e55967411d2033d6fb875f4cf0c7a7c04a3864f4bd.jpg)

![c8adaa32e77623f376c8183e4a7e94094848ad41aca9d5bf1d79a5a7afe27186.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/c8adaa32e77623f376c8183e4a7e94094848ad41aca9d5bf1d79a5a7afe27186.jpg)

![d2d23fb8208a11092138dcda60c6e0de129d3a30dfb80f0aff69684a349ecf35.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/d2d23fb8208a11092138dcda60c6e0de129d3a30dfb80f0aff69684a349ecf35.jpg)

![d835f98b8feff2b23dce5ac6de5f0fb5341c6e575fd800327ef3449bdb648723.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/d835f98b8feff2b23dce5ac6de5f0fb5341c6e575fd800327ef3449bdb648723.jpg)

![dbf3a3c73a0752279731d7d2885efceb8dfbfeebbe49703948ee6075e0172960.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/dbf3a3c73a0752279731d7d2885efceb8dfbfeebbe49703948ee6075e0172960.jpg)

![dc064af78e773cdec62cfbac445c43492488754db772979f830ea030668c888f.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/dc064af78e773cdec62cfbac445c43492488754db772979f830ea030668c888f.jpg)

![e068b34cc4925a413d244f81ac0a300bd03c7f55e33686263f9719fd0590f5ff.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/e068b34cc4925a413d244f81ac0a300bd03c7f55e33686263f9719fd0590f5ff.jpg)

![e2ec07cf4ef79625cf8872879efd6b1591f4aebde6dc874edc59878dc343cfcd.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/e2ec07cf4ef79625cf8872879efd6b1591f4aebde6dc874edc59878dc343cfcd.jpg)

![e30246c3d87438c41e0cc29a37dc32d9b7c0f02828bea0e83bd927fa8a993a11.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/e30246c3d87438c41e0cc29a37dc32d9b7c0f02828bea0e83bd927fa8a993a11.jpg)

![e6a72d611ade5756c5b8cf21353170c74027de46b566c6f5c709def321dbe275.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/e6a72d611ade5756c5b8cf21353170c74027de46b566c6f5c709def321dbe275.jpg)

![f5a5aeca36c31d063e49491c91aefed9405f74f3dacf028c3902f2352745c6a4.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/f5a5aeca36c31d063e49491c91aefed9405f74f3dacf028c3902f2352745c6a4.jpg)

![f6265c9df873aa094a3f0ae648a7ab2fd720dd8ed2ee952295f6951593ad123e.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/images/f6265c9df873aa094a3f0ae648a7ab2fd720dd8ed2ee952295f6951593ad123e.jpg)

### Tables

![5dfe8a2cb614924109169514309a24b0daf8b6fd62ec13e3a3ee8c6c272ba3d2.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/tables/5dfe8a2cb614924109169514309a24b0daf8b6fd62ec13e3a3ee8c6c272ba3d2.jpg)

![868e9bddc7484d2e165e02d6b0325972477751b5bd67da929d5fad8ae7ce8e54.jpg](../icml_results/5133_Are%20Large%20Reasoning%20Models%20Interruptible_/tables/868e9bddc7484d2e165e02d6b0325972477751b5bd67da929d5fad8ae7ce8e54.jpg)

<a id="paper-5086"></a>

## RefineEvo: Planning-Guided Heuristic Evolution with Bidirectional Experience


### Images

![29030786f54ade4e3be3740dd61fc1306e8c785b841e2cf6396b07bea74894e0.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/29030786f54ade4e3be3740dd61fc1306e8c785b841e2cf6396b07bea74894e0.jpg)

![2d8f420f85b4657dd8b34582a72483e75c0115c3ff9472f9642d9ee44edaaa94.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/2d8f420f85b4657dd8b34582a72483e75c0115c3ff9472f9642d9ee44edaaa94.jpg)

![3b08e6fd65eec266ff66f19394c261d1605c314e06ae19a3d20d0d4dc8169526.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/3b08e6fd65eec266ff66f19394c261d1605c314e06ae19a3d20d0d4dc8169526.jpg)

![734726edaeb33fe883c7c54e67b9490aefa4ef2e601b8cd885d7d8ea89b0187a.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/734726edaeb33fe883c7c54e67b9490aefa4ef2e601b8cd885d7d8ea89b0187a.jpg)

![d5bb11ddb4e3ad7edb093f9ebdc8223a4821f464df9f812196c27d665decfda1.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/d5bb11ddb4e3ad7edb093f9ebdc8223a4821f464df9f812196c27d665decfda1.jpg)

![d6f2cb24520e2ec5e31f59892fc333a3d93214459c16b39f27fb5da4e26599ce.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/d6f2cb24520e2ec5e31f59892fc333a3d93214459c16b39f27fb5da4e26599ce.jpg)

![dc28808f6ece867cb01261dd6100597cef3112acba9fd0116d9f94b705c58a50.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/images/dc28808f6ece867cb01261dd6100597cef3112acba9fd0116d9f94b705c58a50.jpg)

### Tables

![01fdf7f636e078b0fb517e9285cce6c40571e56de059dae948bc94d968adef9d.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/01fdf7f636e078b0fb517e9285cce6c40571e56de059dae948bc94d968adef9d.jpg)

![1767410eadd7cc2189e6b9c0157118ea32fb40497c28a38351427ed135bd35c9.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/1767410eadd7cc2189e6b9c0157118ea32fb40497c28a38351427ed135bd35c9.jpg)

![25cc2710cec14b5b3181c55db759900ec7fdcfb8708858bab6beec4c1b01e5f8.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/25cc2710cec14b5b3181c55db759900ec7fdcfb8708858bab6beec4c1b01e5f8.jpg)

![3004ddf6cfb10a8c849c23567ad99ddc83dc482517b63c0852a41388c48880e7.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/3004ddf6cfb10a8c849c23567ad99ddc83dc482517b63c0852a41388c48880e7.jpg)

![3a769e9f7b23d409894c18dd2269cd554a6ea735e419b49a2ffcc53c2defa87a.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/3a769e9f7b23d409894c18dd2269cd554a6ea735e419b49a2ffcc53c2defa87a.jpg)

![44b339d112e9df298dc50d4b83cb427d86bb2c9aab4d56dba19ff13cb885c6c8.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/44b339d112e9df298dc50d4b83cb427d86bb2c9aab4d56dba19ff13cb885c6c8.jpg)

![4e9a430fa2fa0eca8cc017d18a6117901e7372d4935ce16e67f22a0895d63a74.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/4e9a430fa2fa0eca8cc017d18a6117901e7372d4935ce16e67f22a0895d63a74.jpg)

![55473a892bdb0e18251b611c456fb3880c9f6fa931b73932e9860e7a9e8a680b.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/55473a892bdb0e18251b611c456fb3880c9f6fa931b73932e9860e7a9e8a680b.jpg)

![7792603ecdb405c0e8456bea8e88ebe7ddc2ac2bb1229112c3b1bde1c5e0829c.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/7792603ecdb405c0e8456bea8e88ebe7ddc2ac2bb1229112c3b1bde1c5e0829c.jpg)

![9470098463b53f0d3a9ce367d6be9bf273a25b9627daee32cd4a1031ae1f1873.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/9470098463b53f0d3a9ce367d6be9bf273a25b9627daee32cd4a1031ae1f1873.jpg)

![9ffb216c4ecd2f98b75bf728d474c1b1e14e4a4b3e99fdf0b4d35303f0b920d8.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/9ffb216c4ecd2f98b75bf728d474c1b1e14e4a4b3e99fdf0b4d35303f0b920d8.jpg)

![ab122e0fd47d744f4f0ab8ee931b8fc47fa78a0133d45a2604558823a498021f.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/ab122e0fd47d744f4f0ab8ee931b8fc47fa78a0133d45a2604558823a498021f.jpg)

![ad53b3b930b2f716190f991cc2d03b10ec25ea868ad55db592ee4bbb261ab592.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/ad53b3b930b2f716190f991cc2d03b10ec25ea868ad55db592ee4bbb261ab592.jpg)

![ad5a4db8ae05868ad3f53b05bde97667f2e83a51343dcfcc7a65d218f687a52a.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/ad5a4db8ae05868ad3f53b05bde97667f2e83a51343dcfcc7a65d218f687a52a.jpg)

![aff8332230f9b7e59362a94b3fa3a5a43f8b05af44bf1264d428aa4335202c5d.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/aff8332230f9b7e59362a94b3fa3a5a43f8b05af44bf1264d428aa4335202c5d.jpg)

![bb4d1544f4544278ca3b481c065b1aa9b91fb6e75939d17ec76ed36bb7243ff2.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/bb4d1544f4544278ca3b481c065b1aa9b91fb6e75939d17ec76ed36bb7243ff2.jpg)

![bef0c9e50d15260ec0086f54a62ace0863176358302d013c0495b694c2ce974b.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/bef0c9e50d15260ec0086f54a62ace0863176358302d013c0495b694c2ce974b.jpg)

![c48d404c070a1c76ccb1ae49182cead30c2031a2e47d72ebba78874ddd801d2c.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/c48d404c070a1c76ccb1ae49182cead30c2031a2e47d72ebba78874ddd801d2c.jpg)

![c5a0adcf3153e4f86235ad90c684511b318e164aed51e27812e7165e95d149d6.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/c5a0adcf3153e4f86235ad90c684511b318e164aed51e27812e7165e95d149d6.jpg)

![cae48448f220549fd99851f3a2db63da68633848379e9e274c27addd39c5f3f2.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/cae48448f220549fd99851f3a2db63da68633848379e9e274c27addd39c5f3f2.jpg)

![d28a2e58c48921cf0dc5caa8f7278264f233d925d15cfd65914175a18cb02f66.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/d28a2e58c48921cf0dc5caa8f7278264f233d925d15cfd65914175a18cb02f66.jpg)

![d53fca22bcca3d6fa51b0a42acb7d9437a9f4e8f528d34d412eedaa57ea544c3.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/d53fca22bcca3d6fa51b0a42acb7d9437a9f4e8f528d34d412eedaa57ea544c3.jpg)

![da85e97c9ebbbbd0380023e9a6aecdfef2280c2db6c092a314272c34b31c2677.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/da85e97c9ebbbbd0380023e9a6aecdfef2280c2db6c092a314272c34b31c2677.jpg)

![dc920764fc254aa844502e9503ad8ed91774d8bd63fbfd0ef50028ecea8ee068.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/dc920764fc254aa844502e9503ad8ed91774d8bd63fbfd0ef50028ecea8ee068.jpg)

![dd0fc3e4684d051c25b1ae63d0e37141c47d95c983aac1bf459ab52e99c7073c.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/dd0fc3e4684d051c25b1ae63d0e37141c47d95c983aac1bf459ab52e99c7073c.jpg)

![de4b528f460336111e9ab120ddabc67a6e6b3e4a10496be31fb60bcc112ec8a9.jpg](../icml_results/5134_RefineEvo_%20Planning-Guided%20Heuristic%20Evolution%20with%20Bidirectional%20Experience/tables/de4b528f460336111e9ab120ddabc67a6e6b3e4a10496be31fb60bcc112ec8a9.jpg)

<a id="paper-5087"></a>

## AutoWebWorld: Synthesizing Infinite Verifiable Web Environments via Finite State Machines


### Images

![22723982ed013668309667ef8874c38e047f2b18f83630c8a7e2e2a2bf72a7c4.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/22723982ed013668309667ef8874c38e047f2b18f83630c8a7e2e2a2bf72a7c4.jpg)

![3d6a925102bea4fc4a63add60a33ea47c9c7dc7b3b01d2005597eaa1e25c8680.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/3d6a925102bea4fc4a63add60a33ea47c9c7dc7b3b01d2005597eaa1e25c8680.jpg)

![5a38c7b0ec2309ad39766fe56858c9edaec036e70de40bf9278007ec983a2f60.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/5a38c7b0ec2309ad39766fe56858c9edaec036e70de40bf9278007ec983a2f60.jpg)

![9ebb193d555bb1dbd57e28a67d06bc92c08f1be8ea67f045eb66acca177a7da7.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/9ebb193d555bb1dbd57e28a67d06bc92c08f1be8ea67f045eb66acca177a7da7.jpg)

![c3f3313d71d19f3cf32ae08a85c1b7d8c9c9c13ecfdc194d13d78964d62a3b83.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/c3f3313d71d19f3cf32ae08a85c1b7d8c9c9c13ecfdc194d13d78964d62a3b83.jpg)

![daa0b5f40a1b1c66b7716fd65732068d2f8514bec0e4fd013e2041b90c133b71.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/daa0b5f40a1b1c66b7716fd65732068d2f8514bec0e4fd013e2041b90c133b71.jpg)

![f2be14e39ea71c0193590dbc4fffbd3d99e5794807954d284c00b504490eb91a.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/images/f2be14e39ea71c0193590dbc4fffbd3d99e5794807954d284c00b504490eb91a.jpg)

### Tables

![07ab8839e6185974e02fa3178ab635e3eab6e90400a7459d1bce6c76b01b57bd.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/07ab8839e6185974e02fa3178ab635e3eab6e90400a7459d1bce6c76b01b57bd.jpg)

![2aee61058fb68a4335cb18c593ad99ce4c6e93491a6377da44217994fcfdb0bb.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/2aee61058fb68a4335cb18c593ad99ce4c6e93491a6377da44217994fcfdb0bb.jpg)

![546916f75bd990f8bebadcda6f057368ad96813c0f8cf1c2eac5ca36b4e8551a.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/546916f75bd990f8bebadcda6f057368ad96813c0f8cf1c2eac5ca36b4e8551a.jpg)

![6496ca01dfe3e9b1197dde302e5b8a337d8172dc9ff741cc80a3a2b29f5aed84.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/6496ca01dfe3e9b1197dde302e5b8a337d8172dc9ff741cc80a3a2b29f5aed84.jpg)

![6f6549ff4f776fe59dd53b4f0beffdf4f62e3fd58d1e26368de92a5cd517b780.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/6f6549ff4f776fe59dd53b4f0beffdf4f62e3fd58d1e26368de92a5cd517b780.jpg)

![932d0c8856f55318a4598aa929f93311e57d12d6ea816dd1e33b532a373a70c2.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/932d0c8856f55318a4598aa929f93311e57d12d6ea816dd1e33b532a373a70c2.jpg)

![a9cbf6acaf575bb8bb029b787ca5386ecc995cdd612f3abcbdb0538d9d8c1655.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/a9cbf6acaf575bb8bb029b787ca5386ecc995cdd612f3abcbdb0538d9d8c1655.jpg)

![aa1a3740a9083e54984609b2016e671465cafb81c1b5a6fea1e426a63600cbab.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/aa1a3740a9083e54984609b2016e671465cafb81c1b5a6fea1e426a63600cbab.jpg)

![d6afa91e798e4743a7949c3b855fd32cada9ac5248b574a986a8b81edbe0db83.jpg](../icml_results/5135_AutoWebWorld_%20Synthesizing%20Infinite%20Verifiable%20Web%20Environments%20via%20Finite%20State%20Machines/tables/d6afa91e798e4743a7949c3b855fd32cada9ac5248b574a986a8b81edbe0db83.jpg)

<a id="paper-5088"></a>

## Reasoning Cache: Continual Improvement Over Long Horizons via Short-Horizon RL


### Images

![04409f60c8584b216f84c349c5ec25eea96a940e2ab535dcfbadc8b1a0885f3d.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/04409f60c8584b216f84c349c5ec25eea96a940e2ab535dcfbadc8b1a0885f3d.jpg)

![23cd2dd812ed368f66744c63beb5e834511238fcf6da68518f34e5fe6cee0beb.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/23cd2dd812ed368f66744c63beb5e834511238fcf6da68518f34e5fe6cee0beb.jpg)

![2f08b3d80e2acd7455a35a714533afd1c1c4fd8ba7a99d9d27a4d8315ef22209.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/2f08b3d80e2acd7455a35a714533afd1c1c4fd8ba7a99d9d27a4d8315ef22209.jpg)

![65452282ec6b3c820b01c1243810114a6c0d753fcaf646f2e68d71f6a5170ecc.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/65452282ec6b3c820b01c1243810114a6c0d753fcaf646f2e68d71f6a5170ecc.jpg)

![6a17848bbbe1cbc6a969f0ab04816b1f29d314b8198362464290a43f4a1f3af1.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/6a17848bbbe1cbc6a969f0ab04816b1f29d314b8198362464290a43f4a1f3af1.jpg)

![74f2fd01c99ab4bdd094731841212d60ded237f4fd8c13e7d86341f048b69e23.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/74f2fd01c99ab4bdd094731841212d60ded237f4fd8c13e7d86341f048b69e23.jpg)

![9e6a5429871924cf44557c9f9ee43d2e083d358089d17c5e2907812336821ba7.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/9e6a5429871924cf44557c9f9ee43d2e083d358089d17c5e2907812336821ba7.jpg)

![c03cf7cb1bfb4f5c4a3ecffa8e32bff8e785f97b97f0c1be57005836e47e39b1.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/c03cf7cb1bfb4f5c4a3ecffa8e32bff8e785f97b97f0c1be57005836e47e39b1.jpg)

![c86f0c14ef1a001ee6e32821c6bb81bd08b099f7cfb52a3c81033f0e7c09b1a6.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/c86f0c14ef1a001ee6e32821c6bb81bd08b099f7cfb52a3c81033f0e7c09b1a6.jpg)

![ca63e79a6d742070cf1d0c01c28dc425790763b8a217ab5d1cd95c35b8d60815.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/ca63e79a6d742070cf1d0c01c28dc425790763b8a217ab5d1cd95c35b8d60815.jpg)

![d95c7822cf9d70e064f5e1163c8cad0e217a3b314f597fa506362f9e7a9445db.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/d95c7822cf9d70e064f5e1163c8cad0e217a3b314f597fa506362f9e7a9445db.jpg)

![ec0f27becd893f961529de1ac40da6230a0ae7ff63c9f236318369aff8aaa01e.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/ec0f27becd893f961529de1ac40da6230a0ae7ff63c9f236318369aff8aaa01e.jpg)

![f3b7a184474974599c7e228eb8908c0668681907117a6930221d795b0e6d225a.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/f3b7a184474974599c7e228eb8908c0668681907117a6930221d795b0e6d225a.jpg)

![fd4f7879ed6b111721a867934b3d44d7dc2c7466804f73ee1207c5673cfd3033.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/images/fd4f7879ed6b111721a867934b3d44d7dc2c7466804f73ee1207c5673cfd3033.jpg)

### Tables

![2fb8570891f2cb7298a8c71be8492dac9f566544c02ac3bff7d21e46e0841ea3.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/2fb8570891f2cb7298a8c71be8492dac9f566544c02ac3bff7d21e46e0841ea3.jpg)

![4367e63a223b3aeedbfad22f1bd5ae957f25f4b46068da37cfdfb8ed54e8e4f1.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/4367e63a223b3aeedbfad22f1bd5ae957f25f4b46068da37cfdfb8ed54e8e4f1.jpg)

![5a162fbd6bc23a1a3a55d208e302678744621d0f2a4691399caac8e4dd6b5f87.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/5a162fbd6bc23a1a3a55d208e302678744621d0f2a4691399caac8e4dd6b5f87.jpg)

![8db0737dd9237e5eb7e838de4d0ac558fb4a575cc25fc352740ce3c63531301b.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/8db0737dd9237e5eb7e838de4d0ac558fb4a575cc25fc352740ce3c63531301b.jpg)

![bf983307322a354c0b2730a6731a7e31e83fef90d2c6fc3d3deadd63df7423db.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/bf983307322a354c0b2730a6731a7e31e83fef90d2c6fc3d3deadd63df7423db.jpg)

![e47e22e0c14275fee5798b87601c0a972df97ea14442ff619806c0b89c8ddf87.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/e47e22e0c14275fee5798b87601c0a972df97ea14442ff619806c0b89c8ddf87.jpg)

![f20745a37285c79b8e8c0c53af2e384181373bcabcbbeb3208814734fabd0bfb.jpg](../icml_results/5136_Reasoning%20Cache_%20Continual%20Improvement%20Over%20Long%20Horizons%20via%20Short-Horizon%20RL/tables/f20745a37285c79b8e8c0c53af2e384181373bcabcbbeb3208814734fabd0bfb.jpg)

<a id="paper-5089"></a>

## REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large Vision-Language Models


### Images

![0c12620c79dff316d221b3b82da147b89f85e8f06ce4fefdaa3b6e6d0d47ea11.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/0c12620c79dff316d221b3b82da147b89f85e8f06ce4fefdaa3b6e6d0d47ea11.jpg)

![255e2ec9ef44685d478db330c1cf929d4067008bfae6ecfa718e4847ed927e72.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/255e2ec9ef44685d478db330c1cf929d4067008bfae6ecfa718e4847ed927e72.jpg)

![415eeef80e2713f30c96838c1962ca869371c03deb44e0fe0e07f944ee176ee2.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/415eeef80e2713f30c96838c1962ca869371c03deb44e0fe0e07f944ee176ee2.jpg)

![59e16f33a936410d0d79965cfa906740700768aa20daa1d7ab3809878b5e47a0.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/59e16f33a936410d0d79965cfa906740700768aa20daa1d7ab3809878b5e47a0.jpg)

![7a5846edb3e1f794cf0e721f44e9d3331987abd9c3da993b1ad1a26740d01fe9.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/7a5846edb3e1f794cf0e721f44e9d3331987abd9c3da993b1ad1a26740d01fe9.jpg)

![7dc252fdc9b695fde46684b6888da84ab54b95e161ccc7b5cfb2c403bbb43a11.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/7dc252fdc9b695fde46684b6888da84ab54b95e161ccc7b5cfb2c403bbb43a11.jpg)

![92777ffeee8fe2b28325167655c356db81cf4e4ddc233ac7f90df7aebb7b4793.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/92777ffeee8fe2b28325167655c356db81cf4e4ddc233ac7f90df7aebb7b4793.jpg)

![d18b82b88ca76b140b2f0ef62ff512f5fea9b825efb022be86cc5672f32aea29.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/d18b82b88ca76b140b2f0ef62ff512f5fea9b825efb022be86cc5672f32aea29.jpg)

![fc9e936b1d47df35f5782d89211e21f640c8d36bf62cf565c2e2b083cc7a83c4.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/images/fc9e936b1d47df35f5782d89211e21f640c8d36bf62cf565c2e2b083cc7a83c4.jpg)

### Tables

![15c0ee8a50b91b98399c19a43f30922537fcd72b297ab0b1f806ec0b7c869e72.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/15c0ee8a50b91b98399c19a43f30922537fcd72b297ab0b1f806ec0b7c869e72.jpg)

![1f1f92570ccba24965d041249fa207771f7d45d782b7e78fc0d8cf1a9c47025f.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/1f1f92570ccba24965d041249fa207771f7d45d782b7e78fc0d8cf1a9c47025f.jpg)

![24eb3b49b4830273ff7da25cdbe613c95ad7b2ea5d4ea010c8a7770232a6d4b1.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/24eb3b49b4830273ff7da25cdbe613c95ad7b2ea5d4ea010c8a7770232a6d4b1.jpg)

![29f0d007386f8071748ec05af78d7626d6ea5fd34c6853d633943054af7370ec.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/29f0d007386f8071748ec05af78d7626d6ea5fd34c6853d633943054af7370ec.jpg)

![36d0651d15fa4f7ca15defb5773e686647a624cdfd470b28ada71b91ddd34d22.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/36d0651d15fa4f7ca15defb5773e686647a624cdfd470b28ada71b91ddd34d22.jpg)

![36fd5b6caf4e47a083756e2b60cbe72b13c16d45c154ffcd84eb0b2dc6a851d7.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/36fd5b6caf4e47a083756e2b60cbe72b13c16d45c154ffcd84eb0b2dc6a851d7.jpg)

![49fc03cea0591e1b25aa2a01ca444f12cea30ea83b3e3894f4890f51d97f4341.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/49fc03cea0591e1b25aa2a01ca444f12cea30ea83b3e3894f4890f51d97f4341.jpg)

![761e491af84be493e28af6eeab79c540a02fb1a3edd336981ef26443973026a7.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/761e491af84be493e28af6eeab79c540a02fb1a3edd336981ef26443973026a7.jpg)

![7c4e7591df0d7d1ec47ac9e6a2e10da0746ec158830c1d30cd0f70d817096e89.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/7c4e7591df0d7d1ec47ac9e6a2e10da0746ec158830c1d30cd0f70d817096e89.jpg)

![87cf9407d09c5d594e58f383c9920c135770037d5a9c0e92eff5984d13ee0d56.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/87cf9407d09c5d594e58f383c9920c135770037d5a9c0e92eff5984d13ee0d56.jpg)

![9a12346675c0fe506484987dea7010d245be1dafc4256aeb4bf9e02fa182beb4.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/9a12346675c0fe506484987dea7010d245be1dafc4256aeb4bf9e02fa182beb4.jpg)

![ab3bd88fbeabd8c8cf484f084498bc64a4a9a42400e384f9c926fbf527f09d80.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/ab3bd88fbeabd8c8cf484f084498bc64a4a9a42400e384f9c926fbf527f09d80.jpg)

![dc8103be61a2d5d8300909ef118ddd623b836c1c740ed33633d619d9ad9e0332.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/dc8103be61a2d5d8300909ef118ddd623b836c1c740ed33633d619d9ad9e0332.jpg)

![e75e9ea8145d489f61e8e6452994ea3c48316cc204ab2e2e356e3b5280a01998.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/e75e9ea8145d489f61e8e6452994ea3c48316cc204ab2e2e356e3b5280a01998.jpg)

![e993be6c41f8ea3439e4c69ec85a8e2980bf84c56243fe0ec2076693bf10b7ba.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/e993be6c41f8ea3439e4c69ec85a8e2980bf84c56243fe0ec2076693bf10b7ba.jpg)

![f3da0c85da47c07f2ede93bfb2e6f48a21e3c700e6e11b41d770208950a1842d.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/f3da0c85da47c07f2ede93bfb2e6f48a21e3c700e6e11b41d770208950a1842d.jpg)

![f526afbe99aec8d1c142c3f6bdee53143026e601719c5086a7cb6a7272076ad7.jpg](../icml_results/5137_REVIS_%20Sparse%20Latent%20Steering%20to%20Mitigate%20Object%20Hallucination%20in%20Large%20Vision-Language%20Models/tables/f526afbe99aec8d1c142c3f6bdee53143026e601719c5086a7cb6a7272076ad7.jpg)

<a id="paper-5090"></a>

## $f$-Divergence Regularized RLHF: Two Tales of Sampling and Unified Analyses


### Images

![5f01f40c5c6bdaba541a18dc7f7f7072d7da086121c00827c63f478595608b99.jpg](../icml_results/5138_%24f%24-Divergence%20Regularized%20RLHF_%20Two%20Tales%20of%20Sampling%20and%20Unified%20Analyses/images/5f01f40c5c6bdaba541a18dc7f7f7072d7da086121c00827c63f478595608b99.jpg)

![8975a394597d496f7a95bc29e75e76dc70913b32dd80976b2792cd96f89cd582.jpg](../icml_results/5138_%24f%24-Divergence%20Regularized%20RLHF_%20Two%20Tales%20of%20Sampling%20and%20Unified%20Analyses/images/8975a394597d496f7a95bc29e75e76dc70913b32dd80976b2792cd96f89cd582.jpg)

![edbbc346b3ccb2b534bfab4b878cf24c4ea1a0ddd5a8e3b8a1d69f8ea1d648a0.jpg](../icml_results/5138_%24f%24-Divergence%20Regularized%20RLHF_%20Two%20Tales%20of%20Sampling%20and%20Unified%20Analyses/images/edbbc346b3ccb2b534bfab4b878cf24c4ea1a0ddd5a8e3b8a1d69f8ea1d648a0.jpg)

### Tables

![01cb9288f4732ed8d4e7f9e74f81582a31ba375c42221c55cad60952fb868bfc.jpg](../icml_results/5138_%24f%24-Divergence%20Regularized%20RLHF_%20Two%20Tales%20of%20Sampling%20and%20Unified%20Analyses/tables/01cb9288f4732ed8d4e7f9e74f81582a31ba375c42221c55cad60952fb868bfc.jpg)

<a id="paper-5091"></a>

## Genome-Factory: A Library for Tuning, Deploying, and Interpreting Genomic Foundation Models


### Images

![aeaa07408964a3677b02bba647a16f19c079c8124e3ba42827a2de8604061d55.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/images/aeaa07408964a3677b02bba647a16f19c079c8124e3ba42827a2de8604061d55.jpg)

![ce8fe40519165bfb923446ebbba38f2b3ab2abd46b81f1e755c7bfb281567d88.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/images/ce8fe40519165bfb923446ebbba38f2b3ab2abd46b81f1e755c7bfb281567d88.jpg)

### Tables

![35d935dd555178771fed9b15c63c9755bb40bc206be72fd1f843902675e93ad1.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/35d935dd555178771fed9b15c63c9755bb40bc206be72fd1f843902675e93ad1.jpg)

![48cbe3a2dce096598b1ae3e41375c0b669858749878e79c5415a5ef5e9c81f4e.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/48cbe3a2dce096598b1ae3e41375c0b669858749878e79c5415a5ef5e9c81f4e.jpg)

![4ab484ab16638875b89942ba88466375be4f9b58225d3725b1dddaa6864e1ebd.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/4ab484ab16638875b89942ba88466375be4f9b58225d3725b1dddaa6864e1ebd.jpg)

![7aa4ca72cf202759c8abd1c9b6c8bb60216ca37d141483033f751a7792c5e9de.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/7aa4ca72cf202759c8abd1c9b6c8bb60216ca37d141483033f751a7792c5e9de.jpg)

![ad1ed7ca913fec13a12bc5dc7130c771d7425f8c574191fb7969113d1a2b6d03.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/ad1ed7ca913fec13a12bc5dc7130c771d7425f8c574191fb7969113d1a2b6d03.jpg)

![b120bcad89a4859451a8a68af8f6227b8ea35a8ebfed84d38d7efc6c047e6a74.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/b120bcad89a4859451a8a68af8f6227b8ea35a8ebfed84d38d7efc6c047e6a74.jpg)

![d4cd84da96ca15ea53ed641e388ae660eaf2baf9a3ee44607d4e4e7de37069df.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/d4cd84da96ca15ea53ed641e388ae660eaf2baf9a3ee44607d4e4e7de37069df.jpg)

![eb1cdcd86110d5b47f4b8ce14153b2f3e0734ba526c6ce0f48f7c03e3c790f2b.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/eb1cdcd86110d5b47f4b8ce14153b2f3e0734ba526c6ce0f48f7c03e3c790f2b.jpg)

![f2cfbd9e0b553645d46d201ee149a2219d1c9606e0ca7f4cd5d007dafbf9ea6c.jpg](../icml_results/5139_Genome-Factory_%20A%20Library%20for%20Tuning%2C%20Deploying%2C%20and%20Interpreting%20Genomic%20Foundation%20Models/tables/f2cfbd9e0b553645d46d201ee149a2219d1c9606e0ca7f4cd5d007dafbf9ea6c.jpg)

<a id="paper-5092"></a>

## Attention Sinks in Diffusion Transformers: A Causal Analysis


### Images

![27d4886cdd5b94143df6a54d08e47e312dedf209468a0c7746393378b7c8421f.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/27d4886cdd5b94143df6a54d08e47e312dedf209468a0c7746393378b7c8421f.jpg)

![5fb9a2053f36f25c6951aaf8b88f13ea3d3aec3fce19b9de1439783254c7060e.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/5fb9a2053f36f25c6951aaf8b88f13ea3d3aec3fce19b9de1439783254c7060e.jpg)

![8b2cb36ae489ee6bb66e437a47537bf48e09d68073a435cfb7543a9542844d57.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/8b2cb36ae489ee6bb66e437a47537bf48e09d68073a435cfb7543a9542844d57.jpg)

![ea152e34d436be1c0415d3cf20a227c20862187560fbd8faf31e8846c906c9c0.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/ea152e34d436be1c0415d3cf20a227c20862187560fbd8faf31e8846c906c9c0.jpg)

![eed0446677fd5652fe908e821dab27031e63e1c81b4308876a10ee9db269b82a.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/eed0446677fd5652fe908e821dab27031e63e1c81b4308876a10ee9db269b82a.jpg)

![f22484c95422b3190db45495bd86f1ddf4284f98e86706ae568a9638456d02cf.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/f22484c95422b3190db45495bd86f1ddf4284f98e86706ae568a9638456d02cf.jpg)

![f2726b0345b95b32c63fb87a0ab63670e72e3ce0e46aca73830cd26a01682dcd.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/f2726b0345b95b32c63fb87a0ab63670e72e3ce0e46aca73830cd26a01682dcd.jpg)

![f5a84f734375c9ea8d362f4f0f591a70bde9addc0fbadbb1a543af573a59671f.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/images/f5a84f734375c9ea8d362f4f0f591a70bde9addc0fbadbb1a543af573a59671f.jpg)

### Tables

![0435dee0a785ac77577909db15396573630ac6b42ba89ab390944ee8d76fc1c4.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/0435dee0a785ac77577909db15396573630ac6b42ba89ab390944ee8d76fc1c4.jpg)

![126a9125d95ae5a367aaec519662d4eee0c7dfb8a7edd2dc00215057ec29627c.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/126a9125d95ae5a367aaec519662d4eee0c7dfb8a7edd2dc00215057ec29627c.jpg)

![24d77cb0841feac9ade8a97887bd44cd138d9f2e69e9c68f78aa862ede85df6b.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/24d77cb0841feac9ade8a97887bd44cd138d9f2e69e9c68f78aa862ede85df6b.jpg)

![2ebe827ae1ebd7d8424cb6100e78c31be530fdbbd59dce35cb60c552c0c1e1e4.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/2ebe827ae1ebd7d8424cb6100e78c31be530fdbbd59dce35cb60c552c0c1e1e4.jpg)

![3234ebbabe7f87d8b76b6bfe0845df4b8069805506aff4b0003b25de5aaad83e.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/3234ebbabe7f87d8b76b6bfe0845df4b8069805506aff4b0003b25de5aaad83e.jpg)

![45a836707cc08c41636887ef97117c9ed43269322addcba32bafa109afb14061.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/45a836707cc08c41636887ef97117c9ed43269322addcba32bafa109afb14061.jpg)

![50d444b92b9c0cb28e0671a9848d5d7e9652cbf47b4bb42b629e639fe392a159.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/50d444b92b9c0cb28e0671a9848d5d7e9652cbf47b4bb42b629e639fe392a159.jpg)

![722857dcdd1d29dd74f80885c9095dd1b0ae64b4af95059285e9b2d04a5a874b.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/722857dcdd1d29dd74f80885c9095dd1b0ae64b4af95059285e9b2d04a5a874b.jpg)

![7328a1c66883b53148a4336c6c6a2e1c46c0720df43295d30e62b5c97047a60c.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/7328a1c66883b53148a4336c6c6a2e1c46c0720df43295d30e62b5c97047a60c.jpg)

![92b75c291fbf3a07172e0cd044349384e32713e0eb9c1e2a1261f30c73385074.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/92b75c291fbf3a07172e0cd044349384e32713e0eb9c1e2a1261f30c73385074.jpg)

![a49c1ad6a54dae5118275f034b09319399fa88f94410e0817f386d1246cf10af.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/a49c1ad6a54dae5118275f034b09319399fa88f94410e0817f386d1246cf10af.jpg)

![a81c6eb559caa61e749f5e363b4adb8ea5f17925339b6aa00d12a941d4c8b9a5.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/a81c6eb559caa61e749f5e363b4adb8ea5f17925339b6aa00d12a941d4c8b9a5.jpg)

![a9021b8318e1f57e42beb2be4159f3a24e9ffb6351a519d12e0d8aa3e570228d.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/a9021b8318e1f57e42beb2be4159f3a24e9ffb6351a519d12e0d8aa3e570228d.jpg)

![b8b36685b3067de97946817e30f5b295051674f8afe7f20dc564a5d5235db87b.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/b8b36685b3067de97946817e30f5b295051674f8afe7f20dc564a5d5235db87b.jpg)

![bfd86b26725889686f80e99d6e78245a1104b5396e033806c3e16fbd760c9623.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/bfd86b26725889686f80e99d6e78245a1104b5396e033806c3e16fbd760c9623.jpg)

![cd11d9bfb59c17fb2f39c1e3e8b3b7f9450c6e21ace81eaec1da9d27701b4607.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/cd11d9bfb59c17fb2f39c1e3e8b3b7f9450c6e21ace81eaec1da9d27701b4607.jpg)

![cffc79d73ae7529bf00284db728d5b4d52720a3da6c49fb51168264a056a77df.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/cffc79d73ae7529bf00284db728d5b4d52720a3da6c49fb51168264a056a77df.jpg)

![e7872a11f998d94bb1c0620879a01397bf10d37535979b14811373ba426b3fc6.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/e7872a11f998d94bb1c0620879a01397bf10d37535979b14811373ba426b3fc6.jpg)

![eef134dd57210f4f3518f9d36ccb8fb00695bdfbae122a05fda47332b90f1371.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/eef134dd57210f4f3518f9d36ccb8fb00695bdfbae122a05fda47332b90f1371.jpg)

![f2456d7c724901a2344b56f7af633e1ad9cd1bd141522ec4eba5e601cd078000.jpg](../icml_results/5140_Attention%20Sinks%20in%20Diffusion%20Transformers_%20A%20Causal%20Analysis/tables/f2456d7c724901a2344b56f7af633e1ad9cd1bd141522ec4eba5e601cd078000.jpg)

<a id="paper-5093"></a>

## PortraitRL: Reinforcement Learning for Portrait Pose Transfer with Multi-Objective Reward


### Images

![43f032a62f1471c61d311f2a497edc9f60af16c3554140e55842e4a5431c8c88.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/43f032a62f1471c61d311f2a497edc9f60af16c3554140e55842e4a5431c8c88.jpg)

![5c4590104b72196f05249090ed5cd66a0d3c50fd66b8f8ce56b7e92a13a433cf.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/5c4590104b72196f05249090ed5cd66a0d3c50fd66b8f8ce56b7e92a13a433cf.jpg)

![af687fa8f883a28831d9f2ee66175bdab068321aa2fc57f9cf5a75994a887d67.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/af687fa8f883a28831d9f2ee66175bdab068321aa2fc57f9cf5a75994a887d67.jpg)

![b8cd95adc6e53d8d8e59c14145387f95fd8f5133eac4d4c4a3f37cb4e98e1109.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/b8cd95adc6e53d8d8e59c14145387f95fd8f5133eac4d4c4a3f37cb4e98e1109.jpg)

![d36d0ffc35be7e1481ce1b9171bfcdb3e25cdc4449647bb4cfd9e14fa5cce3e5.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/d36d0ffc35be7e1481ce1b9171bfcdb3e25cdc4449647bb4cfd9e14fa5cce3e5.jpg)

![d8c9f3a7a723d426bb3e42037e224f5d1b79d60fab481428cf5e7deee5dd487d.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/d8c9f3a7a723d426bb3e42037e224f5d1b79d60fab481428cf5e7deee5dd487d.jpg)

![e03a06ad2d6b6fa5449ba38b57ac8a92edf2146aec78f4de5defc83121f94f59.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/images/e03a06ad2d6b6fa5449ba38b57ac8a92edf2146aec78f4de5defc83121f94f59.jpg)

### Tables

![373db795c1911195795f39803da81aee9f509ef5d4b037c79592940fe591b1b1.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/373db795c1911195795f39803da81aee9f509ef5d4b037c79592940fe591b1b1.jpg)

![54d8171a7cac33dba75fce0c496cdb1f2dc95ee7a3285b95413352867de61272.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/54d8171a7cac33dba75fce0c496cdb1f2dc95ee7a3285b95413352867de61272.jpg)

![7eeb8a8ef9fd2138917fb7a6dc8476cb415d7d02a12ed2022b687580c2356431.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/7eeb8a8ef9fd2138917fb7a6dc8476cb415d7d02a12ed2022b687580c2356431.jpg)

![8702fad091b2c721189aac21fa460d22e48e4c5f84cfe8d3b75f54b606ebd930.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/8702fad091b2c721189aac21fa460d22e48e4c5f84cfe8d3b75f54b606ebd930.jpg)

![a823f96b47965ece5be6f69e4cc840d5b1e5b162ea0b5659dec44a2f8bfb5e20.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/a823f96b47965ece5be6f69e4cc840d5b1e5b162ea0b5659dec44a2f8bfb5e20.jpg)

![b402bb97c77b84a196026eafe44beb63aa2eb4a8e7fa7937bac8fe3024d25872.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/b402bb97c77b84a196026eafe44beb63aa2eb4a8e7fa7937bac8fe3024d25872.jpg)

![ba159cf56dc24b8dfa894bc9802cfbb94cda2cab44fbe5d3b182ea13009179bc.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/ba159cf56dc24b8dfa894bc9802cfbb94cda2cab44fbe5d3b182ea13009179bc.jpg)

![e51490976f3e19bdcd27be8cec5987a1c1e747a54ec725b96316c101fa8dea42.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/e51490976f3e19bdcd27be8cec5987a1c1e747a54ec725b96316c101fa8dea42.jpg)

![ea1755de97708cb38072ee0437413d3741e569757bc9e50499b758ac9848fdaf.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/ea1755de97708cb38072ee0437413d3741e569757bc9e50499b758ac9848fdaf.jpg)

![f1b108fbd68a627345ee9392baf025bb6260a1636568f707c317e492c59dc36b.jpg](../icml_results/5141_PortraitRL_%20Reinforcement%20Learning%20for%20Portrait%20Pose%20Transfer%20with%20Multi-Objective%20Reward/tables/f1b108fbd68a627345ee9392baf025bb6260a1636568f707c317e492c59dc36b.jpg)

<a id="paper-5094"></a>

## Beyond Majority Voting: Self-Reflective Test-Time Reinforcement Learning for LLM Reasoning


### Images

![34ea56efaa40eb27f9d3cb6cfc8549606e17513557e5b37c6bb3d1ec25247520.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/34ea56efaa40eb27f9d3cb6cfc8549606e17513557e5b37c6bb3d1ec25247520.jpg)

![5c0a5438a8376753c76e24483ac0773701ec5fc2097fcce82dbdbd863c5341ac.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/5c0a5438a8376753c76e24483ac0773701ec5fc2097fcce82dbdbd863c5341ac.jpg)

![71e54d0fd16de7366ab9560aba68bedd75ac4e8b7ed464237d15795277ce58ba.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/71e54d0fd16de7366ab9560aba68bedd75ac4e8b7ed464237d15795277ce58ba.jpg)

![9275945dc44445581595fc655a7b8d8a03e69b04c8cfc635b2136a1504ca8439.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/9275945dc44445581595fc655a7b8d8a03e69b04c8cfc635b2136a1504ca8439.jpg)

![a3bf108d81ef8c289cb08c9735811560578518421864d9d9036db018e995361e.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/images/a3bf108d81ef8c289cb08c9735811560578518421864d9d9036db018e995361e.jpg)

### Tables

![3ccb5634908a273e6217047045f1168bbba54429ce320f838e27aab06f1ceabc.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/3ccb5634908a273e6217047045f1168bbba54429ce320f838e27aab06f1ceabc.jpg)

![56c0388f34e8079645b23582c1690af67cfaadb50a593278e9f80e2a2b2ffd0a.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/56c0388f34e8079645b23582c1690af67cfaadb50a593278e9f80e2a2b2ffd0a.jpg)

![883fb2e160301da507249030fb2c8aa899dae5ca1b1665cf04fca63dc98d9a76.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/883fb2e160301da507249030fb2c8aa899dae5ca1b1665cf04fca63dc98d9a76.jpg)

![d5775a613cd4d136dbf8eca4a09001e2c4e08f0dd1186f1e30bf9b4daf9464d0.jpg](../icml_results/5142_Beyond%20Majority%20Voting_%20Self-Reflective%20Test-Time%20Reinforcement%20Learning%20for%20LLM%20Reasoning/tables/d5775a613cd4d136dbf8eca4a09001e2c4e08f0dd1186f1e30bf9b4daf9464d0.jpg)

<a id="paper-5095"></a>

## From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG


### Images

![3248fc2bb983b328caa63927dde0035813b5ce735efa4753800181b0eb6decde.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/3248fc2bb983b328caa63927dde0035813b5ce735efa4753800181b0eb6decde.jpg)

![3c3721f65f67b0ea2ce8e0e8066cb2cca7f74c70213d00ace3935e624a1e560e.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/3c3721f65f67b0ea2ce8e0e8066cb2cca7f74c70213d00ace3935e624a1e560e.jpg)

![53d4470af51a709ce76908820bcc658b299a5534b482aca0884e7473af1db0ea.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/53d4470af51a709ce76908820bcc658b299a5534b482aca0884e7473af1db0ea.jpg)

![5f28a4636acb6ebf4c161b2232046a17e94ed844f7f799c28b39e437a89d3069.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/5f28a4636acb6ebf4c161b2232046a17e94ed844f7f799c28b39e437a89d3069.jpg)

![a68b538dbf79ed4c67f19307419efd8a6ec48b67502266a60c99d50005570460.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/a68b538dbf79ed4c67f19307419efd8a6ec48b67502266a60c99d50005570460.jpg)

![b4e13321e9cf4fcc97e5a76650672cc8a62df14710bb69b4dac464e33972e771.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/b4e13321e9cf4fcc97e5a76650672cc8a62df14710bb69b4dac464e33972e771.jpg)

![bae207ae1f42500f3bc8dc943288584c8d599ddbb528502c5f7bc233281fd940.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/bae207ae1f42500f3bc8dc943288584c8d599ddbb528502c5f7bc233281fd940.jpg)

![ccb4734ee2952b621f1bf4bc88399bff1bc38be3c51b4bcc6744ab29f4ef0542.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/images/ccb4734ee2952b621f1bf4bc88399bff1bc38be3c51b4bcc6744ab29f4ef0542.jpg)

### Tables

![07c0a9a116c5c13a784de53664a643d5457d08e477a861df777273edc2de2c17.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/07c0a9a116c5c13a784de53664a643d5457d08e477a861df777273edc2de2c17.jpg)

![13fddf8a906b5c75952843005425cacfeb891ac0f99091bfbe70cf56796c60d0.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/13fddf8a906b5c75952843005425cacfeb891ac0f99091bfbe70cf56796c60d0.jpg)

![17e3f0d65158abdc7f5f5c1b2207733d289e5184807af9743452bf0dd281fbd8.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/17e3f0d65158abdc7f5f5c1b2207733d289e5184807af9743452bf0dd281fbd8.jpg)

![29bd647c9add5c56a70e50d2128ba0f6a8e253c3922ba2062bd058651be623cf.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/29bd647c9add5c56a70e50d2128ba0f6a8e253c3922ba2062bd058651be623cf.jpg)

![44022d3b8486524b682a29861335a91ba257476a4c6d3d4066ab870dd5c0b776.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/44022d3b8486524b682a29861335a91ba257476a4c6d3d4066ab870dd5c0b776.jpg)

![76ffaac48e11a6d89650fb16a90344a34ab43150ae70431f3ebb2bfb7eb45bd8.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/76ffaac48e11a6d89650fb16a90344a34ab43150ae70431f3ebb2bfb7eb45bd8.jpg)

![99dac7d9f2e75122ad5a37124a8afc5a3421ecce6ae073cfd114733559b05f16.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/99dac7d9f2e75122ad5a37124a8afc5a3421ecce6ae073cfd114733559b05f16.jpg)

![99f53de6bd105bed06ff17d0d9d3ee051c9f739c8e32152de9072dca62e5e807.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/99f53de6bd105bed06ff17d0d9d3ee051c9f739c8e32152de9072dca62e5e807.jpg)

![9c8c33daec361f1fdf58304f737d7b57a1a4e4f75b94b3b94873822521cba538.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/9c8c33daec361f1fdf58304f737d7b57a1a4e4f75b94b3b94873822521cba538.jpg)

![b70e29884eb4798ceccdf1d6e81799aa455193214c317e4a1005eafee6edfbd7.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/b70e29884eb4798ceccdf1d6e81799aa455193214c317e4a1005eafee6edfbd7.jpg)

![c1611aabe38f353fe16e322bc68776938a2290236398bf1d34d8307fa6b1356b.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/c1611aabe38f353fe16e322bc68776938a2290236398bf1d34d8307fa6b1356b.jpg)

![c4c17b53fdedad3b2d0b6cff83b5b1b0eabfbc8e074d2775baebd29473b20477.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/c4c17b53fdedad3b2d0b6cff83b5b1b0eabfbc8e074d2775baebd29473b20477.jpg)

![cff5d7aadcf264d545c7b043902dffd7a4881d7ffbc46ffc62c217a3d537ec0f.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/cff5d7aadcf264d545c7b043902dffd7a4881d7ffbc46ffc62c217a3d537ec0f.jpg)

![d0a80ae0b2e8096309d11fed7f63eb98e3b2f3253f5e20cbe0254dd3d78d7a36.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/d0a80ae0b2e8096309d11fed7f63eb98e3b2f3253f5e20cbe0254dd3d78d7a36.jpg)

![e3ebcfb80745b74d3c7594dba3bce7a4a8d84fc1f3dcc6c52dc3132b49a02c09.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/e3ebcfb80745b74d3c7594dba3bce7a4a8d84fc1f3dcc6c52dc3132b49a02c09.jpg)

![f7bcd529a63605170de4f44054e6e05dec505f5bed1f47793e683c7e9b410176.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/f7bcd529a63605170de4f44054e6e05dec505f5bed1f47793e683c7e9b410176.jpg)

![fe589e42216c7f7c389c20147f57811ff3706ced837716e5d9e0feefdb7988c4.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/fe589e42216c7f7c389c20147f57811ff3706ced837716e5d9e0feefdb7988c4.jpg)

![fec8045f4e095284226f78b263425e1f63ce6ab4c1aedc638e41995f1ad0b5f4.jpg](../icml_results/5143_From%20Conflict%20to%20Consensus_%20Boosting%20Medical%20Reasoning%20via%20Multi-Round%20Agentic%20RAG/tables/fec8045f4e095284226f78b263425e1f63ce6ab4c1aedc638e41995f1ad0b5f4.jpg)

<a id="paper-5096"></a>

## Score Correction for Generative Models with Probabilistic Constraints


### Images

![06ebcd30df3f330d22960a02560017b2f2ee62b201b1e3577e197628ac36eb69.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/06ebcd30df3f330d22960a02560017b2f2ee62b201b1e3577e197628ac36eb69.jpg)

![167349bedffb818df0b61a4194d9a645b3f17aaac02e3a445e2b44b5059c4575.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/167349bedffb818df0b61a4194d9a645b3f17aaac02e3a445e2b44b5059c4575.jpg)

![1fcde2040fae6e7ab12752f129cb54a35d80f43a5c81533e119db22ae6308379.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/1fcde2040fae6e7ab12752f129cb54a35d80f43a5c81533e119db22ae6308379.jpg)

![2aff7be3e6b3b2d60142311ed879f89de848a0c5b4dd9e7be8ef8a4deb124131.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/2aff7be3e6b3b2d60142311ed879f89de848a0c5b4dd9e7be8ef8a4deb124131.jpg)

![2e002973c16fc000b5a65150df07c527d46866607360f0a260f5ae91f32b1308.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/2e002973c16fc000b5a65150df07c527d46866607360f0a260f5ae91f32b1308.jpg)

![2f785288a3b46d088278991b5215a6fd0822e623cb04b1bb63d85e626411190f.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/2f785288a3b46d088278991b5215a6fd0822e623cb04b1bb63d85e626411190f.jpg)

![40e353366a208b04b713ca5a3dd531ab34d8319156037bff5de8a4f8bfbc1f71.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/40e353366a208b04b713ca5a3dd531ab34d8319156037bff5de8a4f8bfbc1f71.jpg)

![4b3a6f1ee63be2a27f999041477dbb92bd7d78020e863e48b36d86e8af9f1c2a.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/4b3a6f1ee63be2a27f999041477dbb92bd7d78020e863e48b36d86e8af9f1c2a.jpg)

![62bfe6463e50a48a638da2c319533dfd88cc8a9e9619773c3eaa810246f8aa98.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/62bfe6463e50a48a638da2c319533dfd88cc8a9e9619773c3eaa810246f8aa98.jpg)

![6d4df3a8e0bef1b37f03ea0f609f827d26bda6a8a562b01ae4c8faeee19bef1b.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/6d4df3a8e0bef1b37f03ea0f609f827d26bda6a8a562b01ae4c8faeee19bef1b.jpg)

![6dc37d4ad0c71d6bffa3714e8674ef90c23d89fabb79b2a849245870b84b414d.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/6dc37d4ad0c71d6bffa3714e8674ef90c23d89fabb79b2a849245870b84b414d.jpg)

![704df3cb04bcaf3b65a882826f08fbb98d5d7c20f35f3301202f5722028a5868.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/704df3cb04bcaf3b65a882826f08fbb98d5d7c20f35f3301202f5722028a5868.jpg)

![74b3457f2fdac3885809c68eee2a42a21948b57930e5f6f4cc23a8cfb1c4f454.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/74b3457f2fdac3885809c68eee2a42a21948b57930e5f6f4cc23a8cfb1c4f454.jpg)

![7be006308cb093c74b9067e54df3086e47c5373d35ccdc512b1fb3a0ec6af037.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/7be006308cb093c74b9067e54df3086e47c5373d35ccdc512b1fb3a0ec6af037.jpg)

![7cb10a7d6f951cfd580157cb3799e0be0b320dad0eb4fd51a64003cb2a85579a.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/7cb10a7d6f951cfd580157cb3799e0be0b320dad0eb4fd51a64003cb2a85579a.jpg)

![99155558bb72c1e938bfb2605c6e78819d5f244ff51d1c24d983c9ab53669ad7.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/99155558bb72c1e938bfb2605c6e78819d5f244ff51d1c24d983c9ab53669ad7.jpg)

![a84b7eabdcdebd7287e5416129dfff48a25c94edf3c87601161d60d5bc7f2b87.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/a84b7eabdcdebd7287e5416129dfff48a25c94edf3c87601161d60d5bc7f2b87.jpg)

![d224a35a0b89fd5b0e6c0d82f0022a0583bed6af1073196713161f560d2363b8.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/d224a35a0b89fd5b0e6c0d82f0022a0583bed6af1073196713161f560d2363b8.jpg)

![ed7311c9977d8573bd5a515e247f246788f99643f81e4fea8320bdd78a9521d7.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/ed7311c9977d8573bd5a515e247f246788f99643f81e4fea8320bdd78a9521d7.jpg)

![f0e100fed4ef6ed2092d11dffd8530cfbdc5becfb4082dc69d6069e067a98259.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/f0e100fed4ef6ed2092d11dffd8530cfbdc5becfb4082dc69d6069e067a98259.jpg)

![f45e46f9ec8fe605089817c6c676aaf3b2cab4a2a5af57cf29f3bedda1bff832.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/images/f45e46f9ec8fe605089817c6c676aaf3b2cab4a2a5af57cf29f3bedda1bff832.jpg)

### Tables

![5672bb83f82c7af9512acdadafeffed935beb2d4ca2a4860d175281ba68ca0c6.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/tables/5672bb83f82c7af9512acdadafeffed935beb2d4ca2a4860d175281ba68ca0c6.jpg)

![b3d9fc20b41544f1bbb560c315d80f73a6a35136be258c46464a73ad203b7606.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/tables/b3d9fc20b41544f1bbb560c315d80f73a6a35136be258c46464a73ad203b7606.jpg)

![cee8f23c33746e673b04fb09c62fb96f0ce83a6955d8c7d3cbc03c645bc603f8.jpg](../icml_results/5144_Score%20Correction%20for%20Generative%20Models%20with%20Probabilistic%20Constraints/tables/cee8f23c33746e673b04fb09c62fb96f0ce83a6955d8c7d3cbc03c645bc603f8.jpg)

<a id="paper-5097"></a>

## Reasoning Compartmentalization: Bridging the Concretization Gap via Abstraction-based Routing


### Images

![0c7fe6b4c0f144ca14a4a0d6a7afab817b59cd19ab10c165d82dabe130d72225.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/0c7fe6b4c0f144ca14a4a0d6a7afab817b59cd19ab10c165d82dabe130d72225.jpg)

![2ddadc069b06c3e837728537bc08cda3554b9568e3bcdba4a9aa4c90413bd69b.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/2ddadc069b06c3e837728537bc08cda3554b9568e3bcdba4a9aa4c90413bd69b.jpg)

![2f07ac6ce05aa43582f620ba16bbf79770a24a8683665d492ff264751eeaec64.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/2f07ac6ce05aa43582f620ba16bbf79770a24a8683665d492ff264751eeaec64.jpg)

![30ac4b24e7909e34870b0d3c2779557f227a9936efae2186aabcb245d9d16ca1.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/30ac4b24e7909e34870b0d3c2779557f227a9936efae2186aabcb245d9d16ca1.jpg)

![35974df774e3169235bb15c810f645420c22d014a8bfa418da1de90b20aab2ee.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/35974df774e3169235bb15c810f645420c22d014a8bfa418da1de90b20aab2ee.jpg)

![5ed1cadac86c67a64558d9bd006a8420eee641c80e01828ca9d10652b15f17a8.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/5ed1cadac86c67a64558d9bd006a8420eee641c80e01828ca9d10652b15f17a8.jpg)

![728b04e98dfd797fb947ea47e48dc19c0ae894e9ee5a23795c2e5b81f7bf5151.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/728b04e98dfd797fb947ea47e48dc19c0ae894e9ee5a23795c2e5b81f7bf5151.jpg)

![79d39ff1ee971f7401dd55aff7807bf1d75c1cad09cd94008f96a7c2ec2e728d.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/79d39ff1ee971f7401dd55aff7807bf1d75c1cad09cd94008f96a7c2ec2e728d.jpg)

![95ae52ab8a3ce3a40f4e290139a3c29ff6585f919808769f75ddeb96c3dfded2.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/95ae52ab8a3ce3a40f4e290139a3c29ff6585f919808769f75ddeb96c3dfded2.jpg)

![c0a5d9e12d90e09c91772ad0a49e61a74d4095c2ed497c602767e5bb98c8b5e1.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/c0a5d9e12d90e09c91772ad0a49e61a74d4095c2ed497c602767e5bb98c8b5e1.jpg)

![cda3c06f51dcf6c4bd63e3c0a850eaa18fc0e2f28b975473ffb68703a966e75e.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/cda3c06f51dcf6c4bd63e3c0a850eaa18fc0e2f28b975473ffb68703a966e75e.jpg)

![d150cfcaf1ecc00cb6756819ef204a8b07370799fb9b4df6e4a892fadc0b0c9e.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/d150cfcaf1ecc00cb6756819ef204a8b07370799fb9b4df6e4a892fadc0b0c9e.jpg)

![d860044b8cf9299a817477cbf30803e3a9c67c9423fea1bf77497a2cf0d0ea54.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/d860044b8cf9299a817477cbf30803e3a9c67c9423fea1bf77497a2cf0d0ea54.jpg)

![e01d848f1869b147a91684b9b7649a507edbdb3755b216e50b9291677ef99ae2.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/images/e01d848f1869b147a91684b9b7649a507edbdb3755b216e50b9291677ef99ae2.jpg)

### Tables

![3afbc955a32626dfde11d8ad628b8355a232f0c62a6824cae67780394f449f3d.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/3afbc955a32626dfde11d8ad628b8355a232f0c62a6824cae67780394f449f3d.jpg)

![472e8ed5f17a1c1ba9d519bc367958b2da72229ec87a39e06dda86d08a950c07.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/472e8ed5f17a1c1ba9d519bc367958b2da72229ec87a39e06dda86d08a950c07.jpg)

![5d0913fcaaefd304d12c03f749dd17e238743e9c6393e977ecde541b2c201973.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/5d0913fcaaefd304d12c03f749dd17e238743e9c6393e977ecde541b2c201973.jpg)

![7e0e7398185b0b041a36d4cab49bd71bc5683f40287e9804a13359bc35f87f3c.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/7e0e7398185b0b041a36d4cab49bd71bc5683f40287e9804a13359bc35f87f3c.jpg)

![920f7627eb3087893ffee80a8ace83aefb621f7b3ec900492bd0b9526a7b3042.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/920f7627eb3087893ffee80a8ace83aefb621f7b3ec900492bd0b9526a7b3042.jpg)

![d4fdde971b6a12ce62bfa72dbea359e91f405e2d567f35c3304af8c254da3e94.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/d4fdde971b6a12ce62bfa72dbea359e91f405e2d567f35c3304af8c254da3e94.jpg)

![efeba0504eecd9abe9b9fe6b4021bc4fd667761095d5176fbbe79cebcdd1a8ef.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/efeba0504eecd9abe9b9fe6b4021bc4fd667761095d5176fbbe79cebcdd1a8ef.jpg)

![f0c9a41d2c0ed6110e0db5fd7fda3c64d837a8973dfef0ec8f15d2341b2444c7.jpg](../icml_results/5145_Reasoning%20Compartmentalization_%20Bridging%20the%20Concretization%20Gap%20via%20Abstraction-based%20Routing/tables/f0c9a41d2c0ed6110e0db5fd7fda3c64d837a8973dfef0ec8f15d2341b2444c7.jpg)

<a id="paper-5098"></a>

## ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation


### Images

![058df5f4095f7adb3b3055c4b1ec943e2ef65ee115dedf7465ea9c3877a6fd90.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/058df5f4095f7adb3b3055c4b1ec943e2ef65ee115dedf7465ea9c3877a6fd90.jpg)

![05f29ec8bbb25ba34b17cd261e497634a062b2325904fb486c2a6f9686a46c74.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/05f29ec8bbb25ba34b17cd261e497634a062b2325904fb486c2a6f9686a46c74.jpg)

![14ab52e23115330e5eade9a89b9b71e0e992f0ff9a8ce1014387de61125b0a17.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/14ab52e23115330e5eade9a89b9b71e0e992f0ff9a8ce1014387de61125b0a17.jpg)

![201a01add19cccba96587b132de2dff6a7c692722865b9abb1a343d39ac13716.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/201a01add19cccba96587b132de2dff6a7c692722865b9abb1a343d39ac13716.jpg)

![28c569a7815a3956edc8487a1542bd9245b529793cc8d6b48dedf6c00b386900.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/28c569a7815a3956edc8487a1542bd9245b529793cc8d6b48dedf6c00b386900.jpg)

![4a4c4210ce05830109c058392fe0b853f682f77e78937be06d24b9807062dd0a.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/4a4c4210ce05830109c058392fe0b853f682f77e78937be06d24b9807062dd0a.jpg)

![4c974ac096e2577e340d89d139e9134754ac2f51b226044ab890578d947b44a1.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/4c974ac096e2577e340d89d139e9134754ac2f51b226044ab890578d947b44a1.jpg)

![5eaf847fa52df8787fe6e1acb9e2f202f0dfa8b7794643535271ac63a22e34f1.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/5eaf847fa52df8787fe6e1acb9e2f202f0dfa8b7794643535271ac63a22e34f1.jpg)

![5f6e54070bf09fa9ed5f3c6ab98d691bb44c37deb9ae21a0c953543614a86122.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/5f6e54070bf09fa9ed5f3c6ab98d691bb44c37deb9ae21a0c953543614a86122.jpg)

![7ba0e3110d5e9a36c616a707b1abaf9488c712ed757883c5fc78bd1494c32cb7.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/7ba0e3110d5e9a36c616a707b1abaf9488c712ed757883c5fc78bd1494c32cb7.jpg)

![7ebd9b8c7ff3ca98f37a13b61c61b9fea4b28f9eaf4068dafe676a0c3a388462.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/7ebd9b8c7ff3ca98f37a13b61c61b9fea4b28f9eaf4068dafe676a0c3a388462.jpg)

![83a3f5c72b288d7624993deb1b04135c89edf434bb9d41a9d7dd46f2a34beed0.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/83a3f5c72b288d7624993deb1b04135c89edf434bb9d41a9d7dd46f2a34beed0.jpg)

![8798d432bf56f1150882273b6dc9a7bb2f5cc060c752392afbefbcc139d2eeea.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/8798d432bf56f1150882273b6dc9a7bb2f5cc060c752392afbefbcc139d2eeea.jpg)

![8b82da434629e72c8900d6ca9d6c692c5572d0cddadb395a24fc0fb0f19e11a3.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/8b82da434629e72c8900d6ca9d6c692c5572d0cddadb395a24fc0fb0f19e11a3.jpg)

![965e763dfd80db0e10cc02e6d71ce1d79c7ccf3a0d56bfccc9c8f6dc4ed59940.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/965e763dfd80db0e10cc02e6d71ce1d79c7ccf3a0d56bfccc9c8f6dc4ed59940.jpg)

![bf127011fef1885c6307daf70efde9bfac0dc1a6a5ebdc672a7628370898e4cd.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/bf127011fef1885c6307daf70efde9bfac0dc1a6a5ebdc672a7628370898e4cd.jpg)

![c040f47e07b82cccf4c7559f27c61d049277f62429c18ed365b670cc45486507.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/c040f47e07b82cccf4c7559f27c61d049277f62429c18ed365b670cc45486507.jpg)

![d6445671793df98c9efdf2ea9659d45d544844cf91e99885623ce15e9dfca0cf.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/d6445671793df98c9efdf2ea9659d45d544844cf91e99885623ce15e9dfca0cf.jpg)

![ffbb4adf7fd48537a3edcaff21f75320ebb527ba4bdd9c2888f59c6f0bb17c82.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/images/ffbb4adf7fd48537a3edcaff21f75320ebb527ba4bdd9c2888f59c6f0bb17c82.jpg)

### Tables

![2ade56d18d328406484624ede74d20bddceb5868ce3f0160277653bf476d8316.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/2ade56d18d328406484624ede74d20bddceb5868ce3f0160277653bf476d8316.jpg)

![2e9fd221257ee5dac3396fe573a901cdde9680679275d1a508afb7e192a6489a.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/2e9fd221257ee5dac3396fe573a901cdde9680679275d1a508afb7e192a6489a.jpg)

![386c76eb6576c2aff2b1080b3342093a972dbe95f7bbef9e610a0bbde86a36f8.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/386c76eb6576c2aff2b1080b3342093a972dbe95f7bbef9e610a0bbde86a36f8.jpg)

![4476bf380abea7aefcb8a289a542d12bf12916098a48b490cb6f57c895786af8.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/4476bf380abea7aefcb8a289a542d12bf12916098a48b490cb6f57c895786af8.jpg)

![6028f5bc522a3b4936b855cdd932fa67aab899318c4aaf1bdf9e1f5d4001eb9f.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/6028f5bc522a3b4936b855cdd932fa67aab899318c4aaf1bdf9e1f5d4001eb9f.jpg)

![6294004cbba0ef67767baf765eb487781e230c6cb4a4898df53152bf4edcbb24.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/6294004cbba0ef67767baf765eb487781e230c6cb4a4898df53152bf4edcbb24.jpg)

![6bc7527d69704da55e73a6672b49211815c2b587e86e5039d08c56d70563976c.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/6bc7527d69704da55e73a6672b49211815c2b587e86e5039d08c56d70563976c.jpg)

![90d7ddaf28835b49cf264d86d7ed0efa99575a4e39e7788efee7b489f987b09f.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/90d7ddaf28835b49cf264d86d7ed0efa99575a4e39e7788efee7b489f987b09f.jpg)

![a422d9a2832aaf5c24c753f751f958830e6193f6148825779c2cf501a676eed0.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/a422d9a2832aaf5c24c753f751f958830e6193f6148825779c2cf501a676eed0.jpg)

![ce433029eca4a79b261b12f41bf30effd228c2ae7915e0a5e00c6fa38f8bb09d.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/ce433029eca4a79b261b12f41bf30effd228c2ae7915e0a5e00c6fa38f8bb09d.jpg)

![d8701a56eb4edd858e69c161d8c8d39a1c558ee8bce085439afc2a733f9b3a4e.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/d8701a56eb4edd858e69c161d8c8d39a1c558ee8bce085439afc2a733f9b3a4e.jpg)

![dc833a247340c4ae3a1a5600f9b637a8bfa45f7bc8d3b4827b14027b43fae446.jpg](../icml_results/5146_ExCyTIn-Bench_%20Evaluating%20LLM%20agents%20on%20Cyber%20Threat%20Investigation/tables/dc833a247340c4ae3a1a5600f9b637a8bfa45f7bc8d3b4827b14027b43fae446.jpg)

<a id="paper-5099"></a>

## Letting Trajectories Spread: Quality-Preserving Control for Diverse Flow Matching


### Images

![101240709bb03c24cb111fb3404f32a79490402531c07d2731b88c5d824b5fa5.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/101240709bb03c24cb111fb3404f32a79490402531c07d2731b88c5d824b5fa5.jpg)

![1baeed5ee9f02f5c7fa085a0984c493cdcfdabadcbc17c663b0b015a2425dbf4.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/1baeed5ee9f02f5c7fa085a0984c493cdcfdabadcbc17c663b0b015a2425dbf4.jpg)

![1c33958c256373832395f1445e2f11742583d59791f6f757b1290d179f8faab0.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/1c33958c256373832395f1445e2f11742583d59791f6f757b1290d179f8faab0.jpg)

![203ca207c8705c2ceb16e6fbf1eebd006c0959fccaf25c2fdce46d4106affd76.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/203ca207c8705c2ceb16e6fbf1eebd006c0959fccaf25c2fdce46d4106affd76.jpg)

![23e92f887c465bb103dee24a297be634f522c8617564796373a61b659c7d9b2e.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/23e92f887c465bb103dee24a297be634f522c8617564796373a61b659c7d9b2e.jpg)

![3b447512e7585c924bcb25c8f695418d5ff59270a223c6f8d0065fe914021ab8.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/3b447512e7585c924bcb25c8f695418d5ff59270a223c6f8d0065fe914021ab8.jpg)

![3f9e24a0de24dc07c1bd782926eb4445a3271eb633143a9220370fcc11cf786a.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/3f9e24a0de24dc07c1bd782926eb4445a3271eb633143a9220370fcc11cf786a.jpg)

![412f323e6c7d443030c8e8fd86bf488b2333c0bb42079c6c8c48599e08b97254.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/412f323e6c7d443030c8e8fd86bf488b2333c0bb42079c6c8c48599e08b97254.jpg)

![5829b41a1e6e84ae729aa51f11724df78344325715967e92221a4d4d5e5f8c14.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/5829b41a1e6e84ae729aa51f11724df78344325715967e92221a4d4d5e5f8c14.jpg)

![66fcfa0f100b00533ba26c3463b291450038c69553453aea86ceb65753593e4f.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/66fcfa0f100b00533ba26c3463b291450038c69553453aea86ceb65753593e4f.jpg)

![751fe76409570734f0eac8e89650848d93e280da97bf789c6398094accecf84d.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/751fe76409570734f0eac8e89650848d93e280da97bf789c6398094accecf84d.jpg)

![7c7b024b4f58e2f30dff420588a91725a6e1764d2fae90d26c6af3eaecc38c99.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/7c7b024b4f58e2f30dff420588a91725a6e1764d2fae90d26c6af3eaecc38c99.jpg)

![7e29eff7793d6707003dad19738714d178e77956ae20d0e32b326a0d7f2bde10.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/7e29eff7793d6707003dad19738714d178e77956ae20d0e32b326a0d7f2bde10.jpg)

![7e6e2d8c269de330f9527dc085e71b1a71c69e31ce5e2a9c328e4493212fe058.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/7e6e2d8c269de330f9527dc085e71b1a71c69e31ce5e2a9c328e4493212fe058.jpg)

![7f5dacd2023f393030fa0be6b18e1082bf479c95a3a35dc2278fb4962edf775d.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/7f5dacd2023f393030fa0be6b18e1082bf479c95a3a35dc2278fb4962edf775d.jpg)

![83e89280b8d2a2a4a00cef2c57295303152e0f16f623661f2afc8be9e02ad11e.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/83e89280b8d2a2a4a00cef2c57295303152e0f16f623661f2afc8be9e02ad11e.jpg)

![97747a552c01f42ab50f1f0ab13c1f3e0edd52a960652f399c51544a6260cbe5.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/97747a552c01f42ab50f1f0ab13c1f3e0edd52a960652f399c51544a6260cbe5.jpg)

![a6c452b5855116cb8ea4115371b9b7daba8d31ebcff652f4adc7b380ecba2c8a.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/a6c452b5855116cb8ea4115371b9b7daba8d31ebcff652f4adc7b380ecba2c8a.jpg)

![aab26856798c48c61cd5da199a94eb2fe079c9cc2106a8387c750fd8f0dd1b1e.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/aab26856798c48c61cd5da199a94eb2fe079c9cc2106a8387c750fd8f0dd1b1e.jpg)

![b6df21c879dd6518e917bc73ac2b3662af695277f13a3f01191942336e5257c6.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/b6df21c879dd6518e917bc73ac2b3662af695277f13a3f01191942336e5257c6.jpg)

![c39240d05955a97f3116e5d0fe638f94e7b012c113fc3ac2985458a363cceaf2.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/c39240d05955a97f3116e5d0fe638f94e7b012c113fc3ac2985458a363cceaf2.jpg)

![c814d40c0ba2c3dd02c961fec3dbb4700a9d2df6c4f62b6f7099420951c81499.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/c814d40c0ba2c3dd02c961fec3dbb4700a9d2df6c4f62b6f7099420951c81499.jpg)

![cd1653bae13897c327c2a478fbc01aafe69cbbc7e3143986f0a8a5c4e70648a2.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/cd1653bae13897c327c2a478fbc01aafe69cbbc7e3143986f0a8a5c4e70648a2.jpg)

![cf22b840784b4f5720eaafacfb06471c326a7d4fb47ff3af64964f08d799cbc1.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/cf22b840784b4f5720eaafacfb06471c326a7d4fb47ff3af64964f08d799cbc1.jpg)

![d6d0a217a4421f0f9cc0310bdb8c9ac2bba218ffa2a582a96bda86ea143ca637.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/d6d0a217a4421f0f9cc0310bdb8c9ac2bba218ffa2a582a96bda86ea143ca637.jpg)

![d7180034c44e881ab62816d12921448e47af7d0b55a3846a34456c2cc1927f33.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/d7180034c44e881ab62816d12921448e47af7d0b55a3846a34456c2cc1927f33.jpg)

![de4d84ea871acccc72ac626b74170024146ddbd73d03fc335fae1f77e27bdcd9.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/de4d84ea871acccc72ac626b74170024146ddbd73d03fc335fae1f77e27bdcd9.jpg)

![e29cfd81167193cde8d8e77e45d1ed273516da991697d0f6f48164624f4d67cb.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/e29cfd81167193cde8d8e77e45d1ed273516da991697d0f6f48164624f4d67cb.jpg)

![e2c0ba4889b204c962c2a01196fc476504efbae40097ec6cdde5d3d0970e8228.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/e2c0ba4889b204c962c2a01196fc476504efbae40097ec6cdde5d3d0970e8228.jpg)

![f0c1f09e5215a5d4942991fd5a284cd4b010284a646b911818189c7295c7e193.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/f0c1f09e5215a5d4942991fd5a284cd4b010284a646b911818189c7295c7e193.jpg)

![f6ade98c85f1024bc48bc9dd52f4593c0685dd65f16d555df5d1492408fb3c8e.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/images/f6ade98c85f1024bc48bc9dd52f4593c0685dd65f16d555df5d1492408fb3c8e.jpg)

### Tables

![094556c50e7f45865bf312dcd0188e04d89c03ab0087a85b8fba31968b3a3d3c.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/094556c50e7f45865bf312dcd0188e04d89c03ab0087a85b8fba31968b3a3d3c.jpg)

![1d51fc74f8a8b447470b7af63eab0b948ca9448a86290f5b2aac17f9bcac149b.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/1d51fc74f8a8b447470b7af63eab0b948ca9448a86290f5b2aac17f9bcac149b.jpg)

![37eafabe71bbd541c348610b20cd015e66bbec2ca1b15439025697e9925eaa1b.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/37eafabe71bbd541c348610b20cd015e66bbec2ca1b15439025697e9925eaa1b.jpg)

![3ba08f023a2d71912e5fc43d18d5b70eb3a47dafd3e7d5fc0c41614c18ed2f92.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/3ba08f023a2d71912e5fc43d18d5b70eb3a47dafd3e7d5fc0c41614c18ed2f92.jpg)

![3d5977d52d089e2b21f96c13ac6cbb60a6a67b0eafb566f8c2fe426ecf3c5dc4.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/3d5977d52d089e2b21f96c13ac6cbb60a6a67b0eafb566f8c2fe426ecf3c5dc4.jpg)

![4a8113afb16898b4aebc3e56cc4ee4b129fc3998812f557f495130ffb51c8279.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/4a8113afb16898b4aebc3e56cc4ee4b129fc3998812f557f495130ffb51c8279.jpg)

![5741c34b5e46b8bfd79385c8b0344249dce86a7c368dc03b375f0a3618ff5a8c.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/5741c34b5e46b8bfd79385c8b0344249dce86a7c368dc03b375f0a3618ff5a8c.jpg)

![57cabfa97860c4b0b38a3dedaea7915a02fdc53acf9117051b204eb31ee15b25.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/57cabfa97860c4b0b38a3dedaea7915a02fdc53acf9117051b204eb31ee15b25.jpg)

![5d6c74b694f93c97f294864b821ce8418bd34156589ca69926401a8081e00052.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/5d6c74b694f93c97f294864b821ce8418bd34156589ca69926401a8081e00052.jpg)

![66c12c74c6b875c828bb4adfeb9ca41c4fbbef534c67bca653ca08f1949da2cd.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/66c12c74c6b875c828bb4adfeb9ca41c4fbbef534c67bca653ca08f1949da2cd.jpg)

![73404c6928ec5af70129b34d51a43fa6ed9f8136d014d7cb288b1484883c755e.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/73404c6928ec5af70129b34d51a43fa6ed9f8136d014d7cb288b1484883c755e.jpg)

![7c24294a703cc52ce445940efdeb0073025d431dbb32e1ce656b4d92d03bcd8a.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/7c24294a703cc52ce445940efdeb0073025d431dbb32e1ce656b4d92d03bcd8a.jpg)

![81406a9192c0975f9804cf2f32e212a57e0dea411f1b58982f35d0196b71e182.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/81406a9192c0975f9804cf2f32e212a57e0dea411f1b58982f35d0196b71e182.jpg)

![849043442a5582578cff57ef7fbb402ad1f1f17b1844439e113d96c56346c2fc.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/849043442a5582578cff57ef7fbb402ad1f1f17b1844439e113d96c56346c2fc.jpg)

![997b8687de2e2157b6364aef3c9a0a11f9cea8eb31db7207e9c710cc1d20f49c.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/997b8687de2e2157b6364aef3c9a0a11f9cea8eb31db7207e9c710cc1d20f49c.jpg)

![a18c4dc6b6dec1ecc21f3c61c3c0777122fbce45c08f839f094270dcd6e24942.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/a18c4dc6b6dec1ecc21f3c61c3c0777122fbce45c08f839f094270dcd6e24942.jpg)

![a8fbd9622b00ddb85ae0867737334bdaaa74b7e2b92d2cfcd9a9021f86a29f22.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/a8fbd9622b00ddb85ae0867737334bdaaa74b7e2b92d2cfcd9a9021f86a29f22.jpg)

![ada508859a2d2fa1b2aace7623919be8373f499a63024b7bc09a1172b83b6cf1.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/ada508859a2d2fa1b2aace7623919be8373f499a63024b7bc09a1172b83b6cf1.jpg)

![b0dce1b4006f85dfc59e39cc618d8d339f13998d9a588a3eb67bd5f5880307be.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/b0dce1b4006f85dfc59e39cc618d8d339f13998d9a588a3eb67bd5f5880307be.jpg)

![b52f6e69b0668b6e00fd51f7f5fd212f77721012d335e1f34f65cc60581808dc.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/b52f6e69b0668b6e00fd51f7f5fd212f77721012d335e1f34f65cc60581808dc.jpg)

![b5a3baf915beea5bb226d865e1bfa28c86774325f72c559de24b13a66e267900.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/b5a3baf915beea5bb226d865e1bfa28c86774325f72c559de24b13a66e267900.jpg)

![bf08d7a16d477973780428877eef76bc91334e56b3801468fad46c2f2be88b43.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/bf08d7a16d477973780428877eef76bc91334e56b3801468fad46c2f2be88b43.jpg)

![e1aa6ed4af63d78b8b995c52b57b11a6e9f544a9c0e739eb6456c4b2b7896e65.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/e1aa6ed4af63d78b8b995c52b57b11a6e9f544a9c0e739eb6456c4b2b7896e65.jpg)

![f0378e4e77aad5d129303a6f151be351a6984e34c78e3fbc04a81ef80c203d26.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/f0378e4e77aad5d129303a6f151be351a6984e34c78e3fbc04a81ef80c203d26.jpg)

![fc9c28ba5b4e0522162a241489f5ff42233ef859ecaa07df0c76670aafaa2be8.jpg](../icml_results/5147_Letting%20Trajectories%20Spread_%20Quality-Preserving%20Control%20for%20Diverse%20Flow%20Matching/tables/fc9c28ba5b4e0522162a241489f5ff42233ef859ecaa07df0c76670aafaa2be8.jpg)

<a id="paper-5100"></a>

## From Interactions to Principles: Experience-Driven Self-Distillation for Evolving LLM Agents


### Images

![4b7538c12cc3aa010a95cf420c9dd1024d95c2e6457dec34287f57966bd4cd7f.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/images/4b7538c12cc3aa010a95cf420c9dd1024d95c2e6457dec34287f57966bd4cd7f.jpg)

![4d54cbee8d7bb3f6a650f55c9c4a0489c8f817b78342af6ccddfc91d397f1afc.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/images/4d54cbee8d7bb3f6a650f55c9c4a0489c8f817b78342af6ccddfc91d397f1afc.jpg)

![64079e0ffb6a3db818fffd35f9cf4cd25bd8f7343619fba236692852b5b0a59d.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/images/64079e0ffb6a3db818fffd35f9cf4cd25bd8f7343619fba236692852b5b0a59d.jpg)

![d7cf439b6eba8bb0990d0cfbfdb705d365c38709eae8abc606f56ba20dc2fcb0.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/images/d7cf439b6eba8bb0990d0cfbfdb705d365c38709eae8abc606f56ba20dc2fcb0.jpg)

![dff7886cb189e842c8d2f024ffed2e4e56e718b4df3ddc3a9a874ff1f5071bf4.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/images/dff7886cb189e842c8d2f024ffed2e4e56e718b4df3ddc3a9a874ff1f5071bf4.jpg)

### Tables

![00d2399c6a094d4c43ae57763b8668e580a7f9c4fd564d6502493bb5107f8011.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/00d2399c6a094d4c43ae57763b8668e580a7f9c4fd564d6502493bb5107f8011.jpg)

![10df80f8f4562e934dc57a9ae176049b6529753ec36c1e9b84986675fa294f00.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/10df80f8f4562e934dc57a9ae176049b6529753ec36c1e9b84986675fa294f00.jpg)

![716203ad8085ce77071a70336b9f9a503c7f8d1e1a2ee793f1383658100f8b84.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/716203ad8085ce77071a70336b9f9a503c7f8d1e1a2ee793f1383658100f8b84.jpg)

![909d96fd22b03eb6539d998bea281cebfb26344b961d75e078106214c406edb6.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/909d96fd22b03eb6539d998bea281cebfb26344b961d75e078106214c406edb6.jpg)

![94f3cbd0c5f4adba49bb825a092edad831cf4cca1110f37b9558a0908f3b8e99.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/94f3cbd0c5f4adba49bb825a092edad831cf4cca1110f37b9558a0908f3b8e99.jpg)

![9d245e82a3e782cf2cdce2639693deb6af030e4445cd6f43d643d074ed08084b.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/9d245e82a3e782cf2cdce2639693deb6af030e4445cd6f43d643d074ed08084b.jpg)

![a8c131d543bede60d0365025284255ec1696f7d80810e1f2406c9f22e44b8eb5.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/a8c131d543bede60d0365025284255ec1696f7d80810e1f2406c9f22e44b8eb5.jpg)

![bfcc709655ece5df234092708c0e2898f77a81e3458c89fc338f961451a202b8.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/bfcc709655ece5df234092708c0e2898f77a81e3458c89fc338f961451a202b8.jpg)

![c06c4220968dc8645dfd0ad46734896a2e7a5ceb73c6a95ad17cb78b5973ba75.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/c06c4220968dc8645dfd0ad46734896a2e7a5ceb73c6a95ad17cb78b5973ba75.jpg)

![e544ff3985bc432779280417c3400b64d1a7e898c7cbd6949b58c8e6651ad64d.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/e544ff3985bc432779280417c3400b64d1a7e898c7cbd6949b58c8e6651ad64d.jpg)

![ed3ec0977b7c3ee56d93e9a52e7c596c1df26f762e23369fe9b61fcdaa175fd5.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/ed3ec0977b7c3ee56d93e9a52e7c596c1df26f762e23369fe9b61fcdaa175fd5.jpg)

![f8a04b7c9094f3c45aa2f9e243dae9e2b942be8dd036196fc8c9629a263df0ab.jpg](../icml_results/5148_From%20Interactions%20to%20Principles_%20Experience-Driven%20Self-Distillation%20for%20Evolving%20LLM%20Agents/tables/f8a04b7c9094f3c45aa2f9e243dae9e2b942be8dd036196fc8c9629a263df0ab.jpg)

<a id="paper-5101"></a>

## Language Model Networks: Supervision-Efficient Learning through Dense Communication


### Images

![0483bb6ec39ec98b3e9b6a227af4b4858057ff26e149ccee17fb3eb91e51e1a7.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/0483bb6ec39ec98b3e9b6a227af4b4858057ff26e149ccee17fb3eb91e51e1a7.jpg)

![1897a412aed0831f4593fb3639dd528db139f70dd3b3f307a21589929e7eb7d8.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/1897a412aed0831f4593fb3639dd528db139f70dd3b3f307a21589929e7eb7d8.jpg)

![371da7474a6e67103934165db8cc5a2b9f204211dddff2af8d49a56d7568cd52.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/371da7474a6e67103934165db8cc5a2b9f204211dddff2af8d49a56d7568cd52.jpg)

![41c1b95e849330065ea603b9de358f8dbe1403cefd3f7e66b1dfaa0a17e489f5.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/41c1b95e849330065ea603b9de358f8dbe1403cefd3f7e66b1dfaa0a17e489f5.jpg)

![abb029297419059ba53299e11a3c37305630d3a88ac8efc6794ca9d5b3cdced8.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/abb029297419059ba53299e11a3c37305630d3a88ac8efc6794ca9d5b3cdced8.jpg)

![cedd7c2fb6b90b8fe60402b5186c453001f31683345b17ee9d7509324f32707a.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/cedd7c2fb6b90b8fe60402b5186c453001f31683345b17ee9d7509324f32707a.jpg)

![f5b61f67bfaabea5d0f635604f3a3876d88e7b0fc05082fe40c33c0ab6cf0ea6.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/images/f5b61f67bfaabea5d0f635604f3a3876d88e7b0fc05082fe40c33c0ab6cf0ea6.jpg)

### Tables

![10c1658a2066afa08f24641172fab44a714250c2b6649218194ba3d827c5f16b.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/10c1658a2066afa08f24641172fab44a714250c2b6649218194ba3d827c5f16b.jpg)

![374c6b9a76a6bcb079a76e69f418ec543d1a32152f33e422116f48566840d4d4.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/374c6b9a76a6bcb079a76e69f418ec543d1a32152f33e422116f48566840d4d4.jpg)

![3ee271460fab74ee55b0060b3053f12f67595b195218e7fe8bd62be1a520b462.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/3ee271460fab74ee55b0060b3053f12f67595b195218e7fe8bd62be1a520b462.jpg)

![471158004638baaa2ae559a175b45a4f4e3a2b3870b0f8622bc412978f0fa367.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/471158004638baaa2ae559a175b45a4f4e3a2b3870b0f8622bc412978f0fa367.jpg)

![8308e0ac76d234a8960dcc3cf744f1b30a2f3b461c374d7a569b7729cf340d89.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/8308e0ac76d234a8960dcc3cf744f1b30a2f3b461c374d7a569b7729cf340d89.jpg)

![9393ea7a08a11221649c2f86ffe70f98916806f3b54311e44246b871872f2b31.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/9393ea7a08a11221649c2f86ffe70f98916806f3b54311e44246b871872f2b31.jpg)

![9ffd29436b91bddb6203ff43c31f696aa7778b81c1f16cc36affa7ea7828f502.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/9ffd29436b91bddb6203ff43c31f696aa7778b81c1f16cc36affa7ea7828f502.jpg)

![e45db5250f87596f0b5627f67f2f74bb7fbfa688ad6cefac9553564b8d5cc950.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/e45db5250f87596f0b5627f67f2f74bb7fbfa688ad6cefac9553564b8d5cc950.jpg)

![ea76230f8951c2d582ddd7ea94fd01a2bc8385bf81605c86f6850d08f940cf1e.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/ea76230f8951c2d582ddd7ea94fd01a2bc8385bf81605c86f6850d08f940cf1e.jpg)

![fc30dba6f350b2275cbb573097314f8448128c72c488da6679020ff5cc14b595.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/fc30dba6f350b2275cbb573097314f8448128c72c488da6679020ff5cc14b595.jpg)

![fcbabad60eb385d22618ef414212ba0fcd6a9b2701573f2e74e84cc1b669a890.jpg](../icml_results/5149_Language%20Model%20Networks_%20Supervision-Efficient%20Learning%20through%20Dense%20Communication/tables/fcbabad60eb385d22618ef414212ba0fcd6a9b2701573f2e74e84cc1b669a890.jpg)

<a id="paper-5102"></a>

## Time-Series Decomposition as a Standalone Task: A Mechanism-Driven Diagnostic Benchmark


### Images

![01491fab6bb53c00e734f0febae81f2d2187299b84a4da6d96652afabec6f7c3.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/images/01491fab6bb53c00e734f0febae81f2d2187299b84a4da6d96652afabec6f7c3.jpg)

![22be5abb7777972257ca6dcc3da0e88408a4a0c68b70d570752395f0118d8d6b.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/images/22be5abb7777972257ca6dcc3da0e88408a4a0c68b70d570752395f0118d8d6b.jpg)

![cbb91c57e60637bb71a63ed2ed57870e0b8959f9ebe1dfb716442cd9083e82ac.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/images/cbb91c57e60637bb71a63ed2ed57870e0b8959f9ebe1dfb716442cd9083e82ac.jpg)

![e85538b42a6e8ae1784c8866dca19c4ce89567a7e0a873f7ef511b548bd4af12.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/images/e85538b42a6e8ae1784c8866dca19c4ce89567a7e0a873f7ef511b548bd4af12.jpg)

### Tables

![108370505867afcef6191cca3e6b71fc85861a8e2456a8410917cfeb0d577f4a.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/108370505867afcef6191cca3e6b71fc85861a8e2456a8410917cfeb0d577f4a.jpg)

![19c56f8b4e156caeb0b2c2be255867545e82b7800032bcada16f0c005b9b58fa.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/19c56f8b4e156caeb0b2c2be255867545e82b7800032bcada16f0c005b9b58fa.jpg)

![276de9831d59bdd7252acf182d9c0e649fab68728251f91389378bd237eed3a1.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/276de9831d59bdd7252acf182d9c0e649fab68728251f91389378bd237eed3a1.jpg)

![285a995e6a568e86e38825aff771154616230f120e0fe5817ec88bd841dffbf1.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/285a995e6a568e86e38825aff771154616230f120e0fe5817ec88bd841dffbf1.jpg)

![68bec913a0a1474d32ba2fde7525c2cbd93ba9d8e1b55879f5c3bdea1cc268a7.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/68bec913a0a1474d32ba2fde7525c2cbd93ba9d8e1b55879f5c3bdea1cc268a7.jpg)

![7933f21fc51425be386f7f2f5be2b49f180d1994a20a58f2fdb88a4d6f8d476d.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/7933f21fc51425be386f7f2f5be2b49f180d1994a20a58f2fdb88a4d6f8d476d.jpg)

![7dcd20fafba1fb1224ccbb6853f3fcb4089e82043e56a50a34de8d624a80b4e9.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/7dcd20fafba1fb1224ccbb6853f3fcb4089e82043e56a50a34de8d624a80b4e9.jpg)

![c6a0cf5387134dd8cc4c4172befaf9d4af2444a4c8dced0cfc0a57a7063ca3fd.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/c6a0cf5387134dd8cc4c4172befaf9d4af2444a4c8dced0cfc0a57a7063ca3fd.jpg)

![e0b20646d1ab8391da1167d6fb2ec5e08694debd64c2b2cfd18b3e0af963188e.jpg](../icml_results/5150_Time-Series%20Decomposition%20as%20a%20Standalone%20Task_%20A%20Mechanism-Driven%20Diagnostic%20Benchmark/tables/e0b20646d1ab8391da1167d6fb2ec5e08694debd64c2b2cfd18b3e0af963188e.jpg)

<a id="paper-5103"></a>

## Neural–Evolutionary Symbolic Regression with Global Constraints: Constraint-Aware Decoding and Reward Shaping


### Images

![10bdffa2598606c16cd0565f8236cead701a28853f941b9502af1bf191281bcf.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/10bdffa2598606c16cd0565f8236cead701a28853f941b9502af1bf191281bcf.jpg)

![3b9490abfe4ffdcbfd42e015a340d39ff4a8e9e850151f2a1d9f25560123288a.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/3b9490abfe4ffdcbfd42e015a340d39ff4a8e9e850151f2a1d9f25560123288a.jpg)

![60a9895c04af5bd357a7d95c4b56f72b4aa52bdde59ce41b2ace60cd89428499.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/60a9895c04af5bd357a7d95c4b56f72b4aa52bdde59ce41b2ace60cd89428499.jpg)

![6614ce0a1a91677f6f5d00bfdd2e03ec604e7c0c6a05869cf2767b5be87187fc.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/6614ce0a1a91677f6f5d00bfdd2e03ec604e7c0c6a05869cf2767b5be87187fc.jpg)

![71175d9451386625961673eb22c19f6add788b5311c76e515587f85da3d7e41e.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/71175d9451386625961673eb22c19f6add788b5311c76e515587f85da3d7e41e.jpg)

![90ba482d0d6ae0836740da365e0bd5c74808fe608a4249c4f8f2823dcb8cb25b.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/90ba482d0d6ae0836740da365e0bd5c74808fe608a4249c4f8f2823dcb8cb25b.jpg)

![9b72d3959d8e912b1049adc3697142a0718dbbbc5a610f841668a5e940a76012.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/9b72d3959d8e912b1049adc3697142a0718dbbbc5a610f841668a5e940a76012.jpg)

![db34a793bad819ae3076cfbcd79d2cbd5cf15137283913524d6fa983aba329dd.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/images/db34a793bad819ae3076cfbcd79d2cbd5cf15137283913524d6fa983aba329dd.jpg)

### Tables

![19e6e324ad2d0dc2d5e99e03c23ff7b9c53cf71009ea69e4d94076382376a359.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/19e6e324ad2d0dc2d5e99e03c23ff7b9c53cf71009ea69e4d94076382376a359.jpg)

![2e1f9bbc34c431f023ee6e3210b013bf4a69755893e54f7571b36508b914f157.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/2e1f9bbc34c431f023ee6e3210b013bf4a69755893e54f7571b36508b914f157.jpg)

![4939d0edb4a41307eeb9114e6d92b28b236523e9fd8757f890aba645d151e552.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/4939d0edb4a41307eeb9114e6d92b28b236523e9fd8757f890aba645d151e552.jpg)

![4940a13734da9e17a98c7ff4d9abbac2ab11767834466fc3ff76f7f6edcfcd74.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/4940a13734da9e17a98c7ff4d9abbac2ab11767834466fc3ff76f7f6edcfcd74.jpg)

![569f61302d2ba2a388c61feeef2421074f3be231066f2eb4bccb4b8a608ebc5d.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/569f61302d2ba2a388c61feeef2421074f3be231066f2eb4bccb4b8a608ebc5d.jpg)

![87fcd3589677e06c2ce1701a0534d1a36e85b35aa20c9874d0edb784926fdf39.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/87fcd3589677e06c2ce1701a0534d1a36e85b35aa20c9874d0edb784926fdf39.jpg)

![994294cf735c0a441be69a2c6569a7baac42448b368c5e3b4a6c28a2656c7c81.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/994294cf735c0a441be69a2c6569a7baac42448b368c5e3b4a6c28a2656c7c81.jpg)

![a4be1538498420286b2f5239ae0c4440a32573705b690e40618bef0bdabbb277.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/a4be1538498420286b2f5239ae0c4440a32573705b690e40618bef0bdabbb277.jpg)

![a864b339951e94a070e0936c9811d90d0f8ee24f5609dd124a850d01c3d9a86c.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/a864b339951e94a070e0936c9811d90d0f8ee24f5609dd124a850d01c3d9a86c.jpg)

![bbfb5b3ae895181c05f87ab9992a86d9d5078339c3fd10675ed3489625b605a1.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/bbfb5b3ae895181c05f87ab9992a86d9d5078339c3fd10675ed3489625b605a1.jpg)

![bc2f4d0ee5f1de190d2989be93ca67f64c443bd35d73b36b17e581d02c0cc362.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/bc2f4d0ee5f1de190d2989be93ca67f64c443bd35d73b36b17e581d02c0cc362.jpg)

![d4857f81eb7fbb28522f0f450bd273b16162c7e616582fd0dab7823b323c83f4.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/d4857f81eb7fbb28522f0f450bd273b16162c7e616582fd0dab7823b323c83f4.jpg)

![e1bc2d4d5cff485405e37361bc3080e7872942d4e7f28d64a547f330bcd91560.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/e1bc2d4d5cff485405e37361bc3080e7872942d4e7f28d64a547f330bcd91560.jpg)

![ee687327512a0965f8a08df80661f9e00375aba569a50ea5cb3d91296b456156.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/ee687327512a0965f8a08df80661f9e00375aba569a50ea5cb3d91296b456156.jpg)

![fd26fdc0533f2070b9e427c21e17ae77355617357fb37680c7d5940b5bb254d4.jpg](../icml_results/5151_Neural%E2%80%93Evolutionary%20Symbolic%20Regression%20with%20Global%20Constraints_%20Constraint-Aware%20Decoding%20and%20Rewar/tables/fd26fdc0533f2070b9e427c21e17ae77355617357fb37680c7d5940b5bb254d4.jpg)

<a id="paper-5104"></a>

## One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery


### Images

![33cc7156082877623ab00bf5d5911d0bae67be1b3cdbf8edd328137accd689ed.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/images/33cc7156082877623ab00bf5d5911d0bae67be1b3cdbf8edd328137accd689ed.jpg)

![5b7c97cf646ede89b49a07bb7f74ff093022613ff919def8dde92ea1b620896a.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/images/5b7c97cf646ede89b49a07bb7f74ff093022613ff919def8dde92ea1b620896a.jpg)

![abb21925bce4e0085f7f254ff872935952633d9234535e52ce371d3fc8c40aeb.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/images/abb21925bce4e0085f7f254ff872935952633d9234535e52ce371d3fc8c40aeb.jpg)

### Tables

![0791d24d375b9f90d803e42651dd8807d275c891cffe7d2d03201c229ba435bd.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/0791d24d375b9f90d803e42651dd8807d275c891cffe7d2d03201c229ba435bd.jpg)

![24458ed3b4724f7fb8aca2fa105f973e9f847e4364553be736628a368ac7de5b.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/24458ed3b4724f7fb8aca2fa105f973e9f847e4364553be736628a368ac7de5b.jpg)

![3d2990800055e24f635e1f1bbe3193d172eee770cee6a7e48f41d02ebebdfbb4.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/3d2990800055e24f635e1f1bbe3193d172eee770cee6a7e48f41d02ebebdfbb4.jpg)

![4f5ffe56b6dd8990dad622689ad842d89cbfd4cb04552611ba50d1d25d455c13.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/4f5ffe56b6dd8990dad622689ad842d89cbfd4cb04552611ba50d1d25d455c13.jpg)

![6cfaf842d6547e7619b1c96761274af4fc016ab8f4eb0ac53d72ab868c1d2a9e.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/6cfaf842d6547e7619b1c96761274af4fc016ab8f4eb0ac53d72ab868c1d2a9e.jpg)

![7fe7bdb34bb524bbdd8eef611a1679d6f14c761994071ca575421d7948d9151c.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/7fe7bdb34bb524bbdd8eef611a1679d6f14c761994071ca575421d7948d9151c.jpg)

![8ec811dae3e9b50ec5396c9d2d0641e8166594bcfcba4664e9de3801fd2ada9a.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/8ec811dae3e9b50ec5396c9d2d0641e8166594bcfcba4664e9de3801fd2ada9a.jpg)

![97c9aa785a18f8f6b1f82a3da130251276981c6749363490602b03e9c8e9428c.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/97c9aa785a18f8f6b1f82a3da130251276981c6749363490602b03e9c8e9428c.jpg)

![c2e0b8a72250d93a91bfb8703a08bf4dabc0074225a261eb601c5938a36b2c8e.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/c2e0b8a72250d93a91bfb8703a08bf4dabc0074225a261eb601c5938a36b2c8e.jpg)

![ee01feb3694b13759f340ea48418d2fec1ffdcaf285cf4354cad75e91119f6df.jpg](../icml_results/5152_One%20Bug%2C%20Hundreds%20Behind_%20LLMs%20for%20Large-Scale%20Bug%20Discovery/tables/ee01feb3694b13759f340ea48418d2fec1ffdcaf285cf4354cad75e91119f6df.jpg)

<a id="paper-5105"></a>

## When Data Is Scarce: Scaling Sparse Language Models with Repeated Training


### Images

![10d62edf4a04003e2d3ef800003284546fefd98e2c11c86bd86a14107d6ff0b9.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/10d62edf4a04003e2d3ef800003284546fefd98e2c11c86bd86a14107d6ff0b9.jpg)

![31cab1703e89812d116883d41b800fb58fce233dc8c89406e8a5bc0b92ec702a.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/31cab1703e89812d116883d41b800fb58fce233dc8c89406e8a5bc0b92ec702a.jpg)

![4e7bd9fc37a62c7160cb4dea921026009a4fc50c7a3e8f3d77b00799ae23b470.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/4e7bd9fc37a62c7160cb4dea921026009a4fc50c7a3e8f3d77b00799ae23b470.jpg)

![70b276b04ea5322291221879073650dd0af0acbdbdd0f89c3f7d81d686dc2e16.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/70b276b04ea5322291221879073650dd0af0acbdbdd0f89c3f7d81d686dc2e16.jpg)

![70b314e680232be0f409f8d3ba5f09b5ee5aeb5a5970c945314270a1ebfef18b.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/70b314e680232be0f409f8d3ba5f09b5ee5aeb5a5970c945314270a1ebfef18b.jpg)

![bf9910dca4da3dd6aab6e9cacb8cb6e6d26a679754b9bbb9c05c82324a6bb91a.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/bf9910dca4da3dd6aab6e9cacb8cb6e6d26a679754b9bbb9c05c82324a6bb91a.jpg)

![ebb09ff51cb0fdb430bf45d5386c7fe17df069439996ec07f0713fbac43a71d4.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/images/ebb09ff51cb0fdb430bf45d5386c7fe17df069439996ec07f0713fbac43a71d4.jpg)

### Tables

![98d5b204886e251969ba7d3042442edee33f753b7e17212fac02a6f1290c6901.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/tables/98d5b204886e251969ba7d3042442edee33f753b7e17212fac02a6f1290c6901.jpg)

![9f1a9e5ae8e6028ee619821499c98a7ac147de0692dc6b8feb42e435963d128b.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/tables/9f1a9e5ae8e6028ee619821499c98a7ac147de0692dc6b8feb42e435963d128b.jpg)

![c6a8d70a8fae4f99d279ee7e5df2d298c970def66fcdfded0904633d9303df38.jpg](../icml_results/5153_When%20Data%20Is%20Scarce_%20Scaling%20Sparse%20Language%20Models%20with%20Repeated%20Training/tables/c6a8d70a8fae4f99d279ee7e5df2d298c970def66fcdfded0904633d9303df38.jpg)

<a id="paper-5106"></a>

## CoCoEdit: Content-Consistent Image Editing via Region Regularized Reinforcement Learning


### Images

![02f47fcdd4550e2f5b3aca6470620c6dace7601bc98db9755f748bb8040fdf82.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/02f47fcdd4550e2f5b3aca6470620c6dace7601bc98db9755f748bb8040fdf82.jpg)

![07daee9ee210ab3015ad218821a27b7b3ce46d9dae8cfe04c5b5ae238152bc14.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/07daee9ee210ab3015ad218821a27b7b3ce46d9dae8cfe04c5b5ae238152bc14.jpg)

![21b7892dc558d0049812d1c5bbdcde346785eb272767c50d9b949cd2f8ca9272.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/21b7892dc558d0049812d1c5bbdcde346785eb272767c50d9b949cd2f8ca9272.jpg)

![24ecd2c3df0c77d89df3420f26da84497f57a8d1df63403428e3e2bd28961839.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/24ecd2c3df0c77d89df3420f26da84497f57a8d1df63403428e3e2bd28961839.jpg)

![25ac9189054e82d86f1920924771209c38b0d9af71b3af831ae8803c2d5c551b.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/25ac9189054e82d86f1920924771209c38b0d9af71b3af831ae8803c2d5c551b.jpg)

![2726c22d1530e58248ca921c3a0bc06eae72af4a5be5f64e4df5b12f285f6361.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/2726c22d1530e58248ca921c3a0bc06eae72af4a5be5f64e4df5b12f285f6361.jpg)

![64f92ba99d85bea7f2eecc8f73ae2260516e768348874b2f5970ebbc1d72594b.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/64f92ba99d85bea7f2eecc8f73ae2260516e768348874b2f5970ebbc1d72594b.jpg)

![6567929c45de092445bfb535e3a732a1968951fae0e5d298fcc12b95c32c150d.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/6567929c45de092445bfb535e3a732a1968951fae0e5d298fcc12b95c32c150d.jpg)

![7432cf2316f2e2441d6122b95dcbd138adf93a604604b48e9a1d169e8100f268.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/7432cf2316f2e2441d6122b95dcbd138adf93a604604b48e9a1d169e8100f268.jpg)

![8201e57a989625938ba2bc608f70046687e17da25f6b8865e354f327e6b977a7.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/8201e57a989625938ba2bc608f70046687e17da25f6b8865e354f327e6b977a7.jpg)

![a2dd1a200bd5a26dbf3ff5e7c9331d06484fb7668b7f942326ffc939cae349ae.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/a2dd1a200bd5a26dbf3ff5e7c9331d06484fb7668b7f942326ffc939cae349ae.jpg)

![a318ab769b1be757ac5dc652ecc4757acd5f0b28a91dfb9cee2600ca99c6dab0.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/a318ab769b1be757ac5dc652ecc4757acd5f0b28a91dfb9cee2600ca99c6dab0.jpg)

![a4c854c81a9eaded0f85dcb9ddd2d8ac07ae1eb560aaf7fabe47de3a8b84be38.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/a4c854c81a9eaded0f85dcb9ddd2d8ac07ae1eb560aaf7fabe47de3a8b84be38.jpg)

![a7fa5b1b519e856ca4f492b97da38639439c70d37b02d6cd58e9cca3884adc6a.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/a7fa5b1b519e856ca4f492b97da38639439c70d37b02d6cd58e9cca3884adc6a.jpg)

![be0c366b30337b7d14dcae7ba0d29f8784a864925e1bafad481e841b8b2bc7d9.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/be0c366b30337b7d14dcae7ba0d29f8784a864925e1bafad481e841b8b2bc7d9.jpg)

![beabd3517d1bf16c3e34e46ee737923f0b44e0e422cf6186160467982e2550be.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/beabd3517d1bf16c3e34e46ee737923f0b44e0e422cf6186160467982e2550be.jpg)

![cc82598abad9bbc0d3acaf697affffdf3630834cd475be77068ca8ed2d8044dd.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/cc82598abad9bbc0d3acaf697affffdf3630834cd475be77068ca8ed2d8044dd.jpg)

![ccc65c10a5fac49fb8b11821fd053e35ff501a7d24ab18a8e32404f1cdb8454c.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/ccc65c10a5fac49fb8b11821fd053e35ff501a7d24ab18a8e32404f1cdb8454c.jpg)

![e65723df784f256b1b5b100b37ba1c0c68f8f495167fdde439c357c42d023a03.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/e65723df784f256b1b5b100b37ba1c0c68f8f495167fdde439c357c42d023a03.jpg)

![ef8ea4b4231e9eb999cf98e8d0d3b4d34ec309716902295d687b5cadc5f7efd9.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/images/ef8ea4b4231e9eb999cf98e8d0d3b4d34ec309716902295d687b5cadc5f7efd9.jpg)

### Tables

![2cb15253297b2d0e96d5d3979cc9e28c7bde3178d4a19fdaa896e23d875cba90.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/2cb15253297b2d0e96d5d3979cc9e28c7bde3178d4a19fdaa896e23d875cba90.jpg)

![483dd7d29728e50b0b19b6e66cff1778647fda8cab3036122835ae713b58b822.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/483dd7d29728e50b0b19b6e66cff1778647fda8cab3036122835ae713b58b822.jpg)

![5602c5465f2a2c23ff20910ea0770de043c584a28bea038638195fe8de56062a.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/5602c5465f2a2c23ff20910ea0770de043c584a28bea038638195fe8de56062a.jpg)

![74dc23c1f90f604ff862e5220f82762a79f4b29bfeffc5d0a9b259e126885c96.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/74dc23c1f90f604ff862e5220f82762a79f4b29bfeffc5d0a9b259e126885c96.jpg)

![8d76d105da60aaeaece875c8fb437db10683de6496320b1eac5264056c0c1e4e.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/8d76d105da60aaeaece875c8fb437db10683de6496320b1eac5264056c0c1e4e.jpg)

![98bffa70980f494945c9ff2a7e9e364c981861e0ed7026deebc43f68c39710f3.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/98bffa70980f494945c9ff2a7e9e364c981861e0ed7026deebc43f68c39710f3.jpg)

![c114daecaf309b5804812496b9a70489da8e36c3c8e0062ea2941338d7ca8089.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/c114daecaf309b5804812496b9a70489da8e36c3c8e0062ea2941338d7ca8089.jpg)

![d9c26a1aa5e7a2f2246af0d4698044bb6199c84dc7129c3e4bf0c6b51b8f88d6.jpg](../icml_results/5154_CoCoEdit_%20Content-Consistent%20Image%20Editing%20via%20Region%20Regularized%20Reinforcement%20Learning/tables/d9c26a1aa5e7a2f2246af0d4698044bb6199c84dc7129c3e4bf0c6b51b8f88d6.jpg)

<a id="paper-5107"></a>

## HiST: A Hierarchical Sparse Transformer for Cross-Modal Spatial Transcriptomics Modeling


### Images

![017fe301badbe81ea0a145212d1691e32ba0a798a6782a6802d3c987043aa2d4.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/017fe301badbe81ea0a145212d1691e32ba0a798a6782a6802d3c987043aa2d4.jpg)

![03e63b10a72b56a12dd6d66946502b5622619984d926c303b275f0affc6e76f7.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/03e63b10a72b56a12dd6d66946502b5622619984d926c303b275f0affc6e76f7.jpg)

![2998649ff328e2c3386036b77616543be08739f9bd4ba3f1f923d92804a5a2e7.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/2998649ff328e2c3386036b77616543be08739f9bd4ba3f1f923d92804a5a2e7.jpg)

![30bacfaf56277cf3d14f2bcb2d747c57f4991d0908f83a8c4217589bdc47265c.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/30bacfaf56277cf3d14f2bcb2d747c57f4991d0908f83a8c4217589bdc47265c.jpg)

![39e567423d3fb0183da7d916502663f06d88ba0b3c04d7b1befb54cdd66e0fad.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/39e567423d3fb0183da7d916502663f06d88ba0b3c04d7b1befb54cdd66e0fad.jpg)

![439466e362b49f1fc05e3be522ce1e39f24a3037b3f36f8a4df5fd2e5c4809aa.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/439466e362b49f1fc05e3be522ce1e39f24a3037b3f36f8a4df5fd2e5c4809aa.jpg)

![665196e817078ed1fad60da1173b3e5fff1f13dfe4dff4aefe977de406b5c32f.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/665196e817078ed1fad60da1173b3e5fff1f13dfe4dff4aefe977de406b5c32f.jpg)

![6f4fa7fffa21aaae7348af1d7c48728c68beed65e7e31c72a55f87b49172ea9c.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/6f4fa7fffa21aaae7348af1d7c48728c68beed65e7e31c72a55f87b49172ea9c.jpg)

![7df42911de83ddf8e40a426edd3a48c206d31b9b2851822054bce125d1918964.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/7df42911de83ddf8e40a426edd3a48c206d31b9b2851822054bce125d1918964.jpg)

![9b98412639f8663c7cbbb7a5ad757236e49b6dde163a473a94670f1f31586582.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/9b98412639f8663c7cbbb7a5ad757236e49b6dde163a473a94670f1f31586582.jpg)

![a65fa16d04dba4d6aceb6c2a902b4dde05a0112bd2110def90f62e31b8e7ad04.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/a65fa16d04dba4d6aceb6c2a902b4dde05a0112bd2110def90f62e31b8e7ad04.jpg)

![b4c125de8e2b6ef3c53bdca02a09fe849338a0db2a006bd33fe840b306584e90.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/b4c125de8e2b6ef3c53bdca02a09fe849338a0db2a006bd33fe840b306584e90.jpg)

![b6279a46e46f0dae4e7e28b7f9dccf111b7098fb8207024ffd23140a33825fcd.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/b6279a46e46f0dae4e7e28b7f9dccf111b7098fb8207024ffd23140a33825fcd.jpg)

![d76455fe3c6f94d5df1282cec43b4a69b6515a9c888bc71aba3c610eeb9bf1d6.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/d76455fe3c6f94d5df1282cec43b4a69b6515a9c888bc71aba3c610eeb9bf1d6.jpg)

![eb20af0c83381bec0d7534850f648dd1d7ff26b0c419418bf66c3f580e4ce6aa.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/eb20af0c83381bec0d7534850f648dd1d7ff26b0c419418bf66c3f580e4ce6aa.jpg)

![f710d27cc58c87bbb8e25ee8cf6e8c454b3ccbd10a82b5fe2502eb6ad9e149f8.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/f710d27cc58c87bbb8e25ee8cf6e8c454b3ccbd10a82b5fe2502eb6ad9e149f8.jpg)

![f7cce0e6f15be7ef43ae38af1c6adac9a4bc61e45bc80c7cb04f64a73804370d.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/images/f7cce0e6f15be7ef43ae38af1c6adac9a4bc61e45bc80c7cb04f64a73804370d.jpg)

### Tables

![5df17ffd647fed297e0b765b1645a69e977e19427de685ad180af0924ecbc64c.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/tables/5df17ffd647fed297e0b765b1645a69e977e19427de685ad180af0924ecbc64c.jpg)

![727ccbda67a9dbc507ada5ca88f139f5dc51dc862f352ed8c20da875f9b66148.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/tables/727ccbda67a9dbc507ada5ca88f139f5dc51dc862f352ed8c20da875f9b66148.jpg)

![9596334ac5d7a71266d7a208dce98247f19e67fe4956c3853b335ec8a2dfc54d.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/tables/9596334ac5d7a71266d7a208dce98247f19e67fe4956c3853b335ec8a2dfc54d.jpg)

![b1206035b414101e3971fb26ca13b7e1a6b386577e77768e0331753219678f5c.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/tables/b1206035b414101e3971fb26ca13b7e1a6b386577e77768e0331753219678f5c.jpg)

![c2e032550f41a1efffaa126cea2e9c7c4eaafa6265e0de15d7b666e14daf1297.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/tables/c2e032550f41a1efffaa126cea2e9c7c4eaafa6265e0de15d7b666e14daf1297.jpg)

![e75749ae3a940df2c9d84b9ce20f37eecb389803b1ac4853836dff514787ffad.jpg](../icml_results/5155_HiST_%20A%20Hierarchical%20Sparse%20Transformer%20for%20Cross-Modal%20Spatial%20Transcriptomics%20Modeling/tables/e75749ae3a940df2c9d84b9ce20f37eecb389803b1ac4853836dff514787ffad.jpg)

<a id="paper-5108"></a>

## PnP-Corrector: A Universal Correction Framework for Coupled Spatiotemporal Forecasting


### Images

![0c3426d54f8a8f6635dafd018ea818d82d7d60bcb0dfe81dd56fe2b6d7833b00.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/0c3426d54f8a8f6635dafd018ea818d82d7d60bcb0dfe81dd56fe2b6d7833b00.jpg)

![12f085907b95bfb9252c8a9ef3f0a13eb461b8ed854c0964df3322bd1944ad2a.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/12f085907b95bfb9252c8a9ef3f0a13eb461b8ed854c0964df3322bd1944ad2a.jpg)

![20a0bc7ec722ba8d3d930e97f1736f6c228433ad9475dd1b0bb887dac68c761e.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/20a0bc7ec722ba8d3d930e97f1736f6c228433ad9475dd1b0bb887dac68c761e.jpg)

![2374cde6ea069fe5521ffd93aac7ce216e87a38f83a7852669ca184318728e56.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/2374cde6ea069fe5521ffd93aac7ce216e87a38f83a7852669ca184318728e56.jpg)

![2999a6665f997d9e099549b779ff23ebb02dd706e3cc4a03875fdc04d889259d.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/2999a6665f997d9e099549b779ff23ebb02dd706e3cc4a03875fdc04d889259d.jpg)

![4008c1c1d1abcc55df5f1e644984490e34120a9404ff958f0d1ee621f75c3b39.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/4008c1c1d1abcc55df5f1e644984490e34120a9404ff958f0d1ee621f75c3b39.jpg)

![52b8c3e435704699e36b3bb3fc3e329cf62b350b1de1fecadedb675fe64242ea.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/52b8c3e435704699e36b3bb3fc3e329cf62b350b1de1fecadedb675fe64242ea.jpg)

![647b225eff02ef5888d8bde6440176be43f50759e0f12903e3a4dcf3f17b7a76.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/647b225eff02ef5888d8bde6440176be43f50759e0f12903e3a4dcf3f17b7a76.jpg)

![6d385db4bc25bcbed741bbac37a835f8d201bcc4fd52f7e3a22819064eb82e28.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/6d385db4bc25bcbed741bbac37a835f8d201bcc4fd52f7e3a22819064eb82e28.jpg)

![878111704d67ce0d633f8a88dbf29e04c7c2fece8f30313f9a8933469719888a.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/878111704d67ce0d633f8a88dbf29e04c7c2fece8f30313f9a8933469719888a.jpg)

![bfb2031f17aca0ac65854f8964b23c4c7a7aa0a38e1725d20db9bb0935137e24.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/bfb2031f17aca0ac65854f8964b23c4c7a7aa0a38e1725d20db9bb0935137e24.jpg)

![caab67dae7004ba383fccf87a60b8d8b105c79a7ad7cb94b4d6d110e81a1496e.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/caab67dae7004ba383fccf87a60b8d8b105c79a7ad7cb94b4d6d110e81a1496e.jpg)

![e73452995236548bdd3b8aa500d2157cd746481e888c935a82ce0d9eb634403d.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/e73452995236548bdd3b8aa500d2157cd746481e888c935a82ce0d9eb634403d.jpg)

![e752808b3bcd5bab775ea0915d660d1ffb4a13ecd02640275e77ac18a694e5ac.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/e752808b3bcd5bab775ea0915d660d1ffb4a13ecd02640275e77ac18a694e5ac.jpg)

![eaf14b4b2db1fb99972762904d7571affa854aa67227efc8264a22db4bb5ed1c.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/images/eaf14b4b2db1fb99972762904d7571affa854aa67227efc8264a22db4bb5ed1c.jpg)

### Tables

![1771ff621c3eed4d2ad77d1b0be166158793db6db390a1d97e2206c155afa0be.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/1771ff621c3eed4d2ad77d1b0be166158793db6db390a1d97e2206c155afa0be.jpg)

![188a64471d27ad190f980758c8f095616867037022dc542f2207dcf998964c8f.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/188a64471d27ad190f980758c8f095616867037022dc542f2207dcf998964c8f.jpg)

![21d87ea93e5612ea2af5a9a568db346911573f9f465f0f33b8a09d1fc04f20fb.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/21d87ea93e5612ea2af5a9a568db346911573f9f465f0f33b8a09d1fc04f20fb.jpg)

![32834ac1c47eaef2e9fe4899a8481464873a73c4ba01760a6ce1dc43084402c6.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/32834ac1c47eaef2e9fe4899a8481464873a73c4ba01760a6ce1dc43084402c6.jpg)

![53d82ba1a8d0d2b0ebe2cfa497ce4a57c8d0247f5f73c789bf599a6b7969064c.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/53d82ba1a8d0d2b0ebe2cfa497ce4a57c8d0247f5f73c789bf599a6b7969064c.jpg)

![baa12400bd12b835771874698657faad3a98f9ce52326de41aa31131eecda191.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/baa12400bd12b835771874698657faad3a98f9ce52326de41aa31131eecda191.jpg)

![c15a8c3efafc84d833c370867dab94085b131f5b9963e5b4ead7f689b11f92ff.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/c15a8c3efafc84d833c370867dab94085b131f5b9963e5b4ead7f689b11f92ff.jpg)

![d2da1de2a08f1da6c26d4cf364851882884375cdb082fd49c56e3d778f0d0dd0.jpg](../icml_results/5156_PnP-Corrector_%20A%20Universal%20Correction%20Framework%20for%20Coupled%20Spatiotemporal%20Forecasting/tables/d2da1de2a08f1da6c26d4cf364851882884375cdb082fd49c56e3d778f0d0dd0.jpg)

<a id="paper-5109"></a>

## From Interaction Trajectories to Prompt Rules: Credit Assignment for Multi-Agent Prompt Optimization


### Images

![2a9c791d371fee6f4a2bcb9e1d82f6ba00910528826c44d5d7daaa6d6460c7b0.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/2a9c791d371fee6f4a2bcb9e1d82f6ba00910528826c44d5d7daaa6d6460c7b0.jpg)

![338ab45b70b58e3d481a5d7a7bd5acc3b85576fedd0b855d5b8e5c6255760561.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/338ab45b70b58e3d481a5d7a7bd5acc3b85576fedd0b855d5b8e5c6255760561.jpg)

![418612a921f8425cbcf593aadc225fd070d4f54548fdf0cf61640765f611df73.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/418612a921f8425cbcf593aadc225fd070d4f54548fdf0cf61640765f611df73.jpg)

![8ab2aa4629e599230625a8684f99043b2750a0af236f63f39441c76d6b1545d7.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/8ab2aa4629e599230625a8684f99043b2750a0af236f63f39441c76d6b1545d7.jpg)

![92ef96eb32bb2daafe759c6da39259112dd4b6d67d46e1fb505b89b207bb608c.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/92ef96eb32bb2daafe759c6da39259112dd4b6d67d46e1fb505b89b207bb608c.jpg)

![ab74e425124fcb010fdcb4990a2ee81af2acff2404e24215f88c544c33c332e2.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/ab74e425124fcb010fdcb4990a2ee81af2acff2404e24215f88c544c33c332e2.jpg)

![e42cefc186353a1698bee3e70172ab4d09de6bd55cbf3837f789ae88648ef3b4.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/images/e42cefc186353a1698bee3e70172ab4d09de6bd55cbf3837f789ae88648ef3b4.jpg)

### Tables

![1a0f70cedc842a2f74057fc48a57e90d02a4aaee4914750260c1258ac9f5fa64.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/1a0f70cedc842a2f74057fc48a57e90d02a4aaee4914750260c1258ac9f5fa64.jpg)

![205eaf6f81416f74c806d05f747255619b410819ff721a7c3d5afd7347dbbd03.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/205eaf6f81416f74c806d05f747255619b410819ff721a7c3d5afd7347dbbd03.jpg)

![6618bb34d4f76002cf7abe6b6f45a5a3141fa296a15887e86e73116c11c0c8cf.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/6618bb34d4f76002cf7abe6b6f45a5a3141fa296a15887e86e73116c11c0c8cf.jpg)

![69b03e33b7d392dc94f839e7cb1d1bf9a4df463859416c5b38f6623495716cd4.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/69b03e33b7d392dc94f839e7cb1d1bf9a4df463859416c5b38f6623495716cd4.jpg)

![aa9104264b34bcab3e85353faec0bb58ddca4318ce01475677f42a2dd0e7267d.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/aa9104264b34bcab3e85353faec0bb58ddca4318ce01475677f42a2dd0e7267d.jpg)

![bdcaa5e7ecd00a55ddbcbce0efe6c8aee7a1abffa502c6393a1a1fc19b352f16.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/bdcaa5e7ecd00a55ddbcbce0efe6c8aee7a1abffa502c6393a1a1fc19b352f16.jpg)

![c72e4027dcdf94b9293d24e27f88733ddd34ba2fd33a7c681a5a21c797dc7fa1.jpg](../icml_results/5157_From%20Interaction%20Trajectories%20to%20Prompt%20Rules_%20Credit%20Assignment%20for%20Multi-Agent%20Prompt%20Optimization/tables/c72e4027dcdf94b9293d24e27f88733ddd34ba2fd33a7c681a5a21c797dc7fa1.jpg)

<a id="paper-5110"></a>

## Proteo-R1: Reasoning Foundation Models for De Novo Protein Design


### Images

![4a7739367876e848231dbd14aeca73918073b88a4a2009e4e74d69d1b7ff1600.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/images/4a7739367876e848231dbd14aeca73918073b88a4a2009e4e74d69d1b7ff1600.jpg)

![af7ceb87190454986bb5c5d763b3a1fa2348f78f546a35064c674ea6507ea5dc.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/images/af7ceb87190454986bb5c5d763b3a1fa2348f78f546a35064c674ea6507ea5dc.jpg)

### Tables

![0660c14a4780002544971e7ee912a0d8517f6f1b99298d51441af4ced6f25011.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/0660c14a4780002544971e7ee912a0d8517f6f1b99298d51441af4ced6f25011.jpg)

![10cb0394530d8dbe6a98b905aeeda5379776db106e27e080170e964c2626429b.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/10cb0394530d8dbe6a98b905aeeda5379776db106e27e080170e964c2626429b.jpg)

![25d6b33a43226a548d7c0f1940472f3d3f635c762e6a2cba04f0b655a36fa229.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/25d6b33a43226a548d7c0f1940472f3d3f635c762e6a2cba04f0b655a36fa229.jpg)

![5c71a3e13b57ef08e6c0dd8a1950847a3396c7e62239e8c65b3d21f2f0da910d.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/5c71a3e13b57ef08e6c0dd8a1950847a3396c7e62239e8c65b3d21f2f0da910d.jpg)

![7308b7b076f2f63ac69e2081a84f922e080bdc5d1687936412ada186712e1fc6.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/7308b7b076f2f63ac69e2081a84f922e080bdc5d1687936412ada186712e1fc6.jpg)

![880ea6610d21d9213cf0c0c814ba17d8ea0af7b164fc7e4c468b0fc561054012.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/880ea6610d21d9213cf0c0c814ba17d8ea0af7b164fc7e4c468b0fc561054012.jpg)

![8816d0f3bf20ceb716c3313a86e0fa0243345bc5d366f4be937f43f66172ab24.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/8816d0f3bf20ceb716c3313a86e0fa0243345bc5d366f4be937f43f66172ab24.jpg)

![952b4fba0434076919e27b77dacf3ad0ba7a3b1252641226895d0af53aa65eb7.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/952b4fba0434076919e27b77dacf3ad0ba7a3b1252641226895d0af53aa65eb7.jpg)

![b3c8f186d8a3296715eb819bc56dfd9974428523884c4fc5f182b4b76d07a2ef.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/b3c8f186d8a3296715eb819bc56dfd9974428523884c4fc5f182b4b76d07a2ef.jpg)

![c855c31164cdec79ca43c7c69025ae5938f0bbe7952f99782ef5ccc180f14618.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/c855c31164cdec79ca43c7c69025ae5938f0bbe7952f99782ef5ccc180f14618.jpg)

![d5e3c4bb464f16cf77afc39c515bffb5d62612a39f0d01d576b371b66e166ea7.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/d5e3c4bb464f16cf77afc39c515bffb5d62612a39f0d01d576b371b66e166ea7.jpg)

![d9671eb3d28da39ecfed9cf1e53f211f193284da10ce1e9aacfb5ce00b90660d.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/d9671eb3d28da39ecfed9cf1e53f211f193284da10ce1e9aacfb5ce00b90660d.jpg)

![e0d0e0b20cd76dc295409a12cd8d732c542aabab7f77d0b326155245fb9eb302.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/e0d0e0b20cd76dc295409a12cd8d732c542aabab7f77d0b326155245fb9eb302.jpg)

![e529cfa066f1dc129996d75c173e368c17ac15d0ebef0062f85e86263c8f1ec6.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/e529cfa066f1dc129996d75c173e368c17ac15d0ebef0062f85e86263c8f1ec6.jpg)

![efedd5360e3d018d6ca74b1a19dae7587ba4072b5e3be7e72e02607b810c7e00.jpg](../icml_results/5158_Proteo-R1_%20Reasoning%20Foundation%20Models%20for%20De%20Novo%20Protein%20Design/tables/efedd5360e3d018d6ca74b1a19dae7587ba4072b5e3be7e72e02607b810c7e00.jpg)

<a id="paper-5111"></a>

## Trajectory-Aware Certified Decentralized Unlearning via SGD Stability


### Images

![0e34d887219bb63b99429356d4e74bac14b4937d822ca1648b76139772a592ca.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/0e34d887219bb63b99429356d4e74bac14b4937d822ca1648b76139772a592ca.jpg)

![0faca0f115f724563186e4c32d032c7aabee5482c2a9b22542eb47e285b566fa.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/0faca0f115f724563186e4c32d032c7aabee5482c2a9b22542eb47e285b566fa.jpg)

![19a2fa7c86e3de63399d745843c3c50793867fd082cfa5814d77884fdbfa4985.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/19a2fa7c86e3de63399d745843c3c50793867fd082cfa5814d77884fdbfa4985.jpg)

![5459f6384a54c9dfc7d3522e8ba04863774d2e9d04c0b10d61637abb4309c3e5.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/5459f6384a54c9dfc7d3522e8ba04863774d2e9d04c0b10d61637abb4309c3e5.jpg)

![6ac00bac5a3a76e3b18434ecd6dcf91f14bac285a60b065c46eb4d5c078acf5d.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/6ac00bac5a3a76e3b18434ecd6dcf91f14bac285a60b065c46eb4d5c078acf5d.jpg)

![7995e1fd66161bac2b94e7d217c9f34221e38d03b16b6fbbd537bdd18cb7cd02.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/7995e1fd66161bac2b94e7d217c9f34221e38d03b16b6fbbd537bdd18cb7cd02.jpg)

![7c4adec61756d422aaff7b2b4582c3780bf6c6a4e729ae6c396904c79427fed5.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/7c4adec61756d422aaff7b2b4582c3780bf6c6a4e729ae6c396904c79427fed5.jpg)

![a20a1a89d320a069a450bf9ced3f9ee63bc8316170a907b97920d9a0d7dbba58.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/a20a1a89d320a069a450bf9ced3f9ee63bc8316170a907b97920d9a0d7dbba58.jpg)

![aae1386ca70155bc3208113dabfbf1b930f697bab877775e1c79a1ebabfc1ced.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/aae1386ca70155bc3208113dabfbf1b930f697bab877775e1c79a1ebabfc1ced.jpg)

![cd19eed0350a7d26182adabcb0c1a443dd300d9a0b71189525d6d9c9c2d73f63.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/cd19eed0350a7d26182adabcb0c1a443dd300d9a0b71189525d6d9c9c2d73f63.jpg)

![e32b4b7daa8277d46725965c88b47f83328e011196688cb3ee48269e1d0227df.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/e32b4b7daa8277d46725965c88b47f83328e011196688cb3ee48269e1d0227df.jpg)

![fea155cecac2967b4a6c2e02cd8a1e8c0aef6f2aa43d16b62a5be1194bcf5554.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/images/fea155cecac2967b4a6c2e02cd8a1e8c0aef6f2aa43d16b62a5be1194bcf5554.jpg)

### Tables

![3f17cd0996d9dd1ad9466f549412a398d5c0378e2ca70e6e3a5864a0951b29e6.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/3f17cd0996d9dd1ad9466f549412a398d5c0378e2ca70e6e3a5864a0951b29e6.jpg)

![509b89f39bde27be2aa459fde52264e3d3643ee9534a8156f8c76f00f7e72217.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/509b89f39bde27be2aa459fde52264e3d3643ee9534a8156f8c76f00f7e72217.jpg)

![50e23f9c1a64138a8a79225b035aea198337ab5819242d5e01dd80d7ccf8425d.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/50e23f9c1a64138a8a79225b035aea198337ab5819242d5e01dd80d7ccf8425d.jpg)

![52c6d71997bdf6ec3af475198f3d019abb286d4a40ec416c699d010253ffa568.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/52c6d71997bdf6ec3af475198f3d019abb286d4a40ec416c699d010253ffa568.jpg)

![56c4cbb216cfe41231605fd9ae8a24867664aa6ed04c739bb65b3c38f91afe8a.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/56c4cbb216cfe41231605fd9ae8a24867664aa6ed04c739bb65b3c38f91afe8a.jpg)

![6a99ee273f1d393ea57e2f334f013f776633fdbcd1f0fafde9cd3ae5e20124ac.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/6a99ee273f1d393ea57e2f334f013f776633fdbcd1f0fafde9cd3ae5e20124ac.jpg)

![98bd17f545ba1dcf6f5e55e500efb942db2c0f4ce8cd3ab49531a15c82cd9809.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/98bd17f545ba1dcf6f5e55e500efb942db2c0f4ce8cd3ab49531a15c82cd9809.jpg)

![9fdfc47164a9ab0730c7e3ab83990335635fe8ba9afdaf13b6a518fb4f3162e1.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/9fdfc47164a9ab0730c7e3ab83990335635fe8ba9afdaf13b6a518fb4f3162e1.jpg)

![d660dcc40693be81d66578e8f755a230de0eb2bb5a2c0cf057982e25481ed558.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/d660dcc40693be81d66578e8f755a230de0eb2bb5a2c0cf057982e25481ed558.jpg)

![fc3a63ca6aaa0df4d09144237b2f52a30b3331bc9c2a95cc1cba18235c2cb771.jpg](../icml_results/5159_Trajectory-Aware%20Certified%20Decentralized%20Unlearning%20via%20SGD%20Stability/tables/fc3a63ca6aaa0df4d09144237b2f52a30b3331bc9c2a95cc1cba18235c2cb771.jpg)

<a id="paper-5112"></a>

## Why Self-Distillation Helps and Hurts: Denoising vs. Signal Forgetting


### Images

![25f98366c7ace319765bf867b5e866aa93d6fbd36630d1bc820dac6c3d67462a.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/25f98366c7ace319765bf867b5e866aa93d6fbd36630d1bc820dac6c3d67462a.jpg)

![3b69458e9750be861ced2ca9acc5029f53588fb5ff036dd7766336f8a7ddd313.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/3b69458e9750be861ced2ca9acc5029f53588fb5ff036dd7766336f8a7ddd313.jpg)

![56dda979791c2250d8e2aa134efd0c80e77d04631c05bee7abbd25fb846c06d0.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/56dda979791c2250d8e2aa134efd0c80e77d04631c05bee7abbd25fb846c06d0.jpg)

![83315799dd1fa03b55477937a75cfb5f4d57b05ead709d7ed848db6316d2295c.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/83315799dd1fa03b55477937a75cfb5f4d57b05ead709d7ed848db6316d2295c.jpg)

![83b81603c3cc0169b353c0f1dc582947a69a94682d6eed1952c0ebe06f421058.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/83b81603c3cc0169b353c0f1dc582947a69a94682d6eed1952c0ebe06f421058.jpg)

![9a43c8218030405062008375688135dad0544afb54a766beda38d02ea36fee55.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/9a43c8218030405062008375688135dad0544afb54a766beda38d02ea36fee55.jpg)

![a41f731e923299e2004bd71b46acff77231e70bba7527fc70fa38462b7492566.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/a41f731e923299e2004bd71b46acff77231e70bba7527fc70fa38462b7492566.jpg)

![dc386ead9b14e2094bc4e8446982e961c4b25d62812be472ddc956d21c5dbdf1.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/dc386ead9b14e2094bc4e8446982e961c4b25d62812be472ddc956d21c5dbdf1.jpg)

![f850f4d0dfa7bc346808d22db06234034c5487fad1f0a7683d2e6f4900e35998.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/f850f4d0dfa7bc346808d22db06234034c5487fad1f0a7683d2e6f4900e35998.jpg)

![fd8aac53c38657771d5f1ea54239853e9698a1825440c781b7e2a1ce87fbb2f7.jpg](../icml_results/5160_Why%20Self-Distillation%20Helps%20and%20Hurts_%20Denoising%20vs.%20Signal%20Forgetting/images/fd8aac53c38657771d5f1ea54239853e9698a1825440c781b7e2a1ce87fbb2f7.jpg)

<a id="paper-5113"></a>

## Speedup Patch: Learning a Plug-and-Play Policy to Accelerate Embodied Manipulation


### Images

![03dc2fc1b5c27ce047fa32f3861cf189f7d79fde8aa2a2e04939372e36423eeb.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/03dc2fc1b5c27ce047fa32f3861cf189f7d79fde8aa2a2e04939372e36423eeb.jpg)

![04fc29cf1687df7c0a165c92c40c5a446d986853526d0e0419810de31beb8e06.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/04fc29cf1687df7c0a165c92c40c5a446d986853526d0e0419810de31beb8e06.jpg)

![1d53ae2e1077e351a068ae94b94ab806d9f8e11d0c1ffbbd05cc966cff2c3658.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/1d53ae2e1077e351a068ae94b94ab806d9f8e11d0c1ffbbd05cc966cff2c3658.jpg)

![22b5e26092f59a917b84a427ef959b0b9d67dd47a9e0718578e523b44634a6c0.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/22b5e26092f59a917b84a427ef959b0b9d67dd47a9e0718578e523b44634a6c0.jpg)

![336a1d3991b99f6d497365ef12594f0f7ca316a207c925badb775ac0127c5c6f.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/336a1d3991b99f6d497365ef12594f0f7ca316a207c925badb775ac0127c5c6f.jpg)

![3673ba4c8c281494d31ea3ba39873b1ef695791e9900cdb33f8a761eef242859.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/3673ba4c8c281494d31ea3ba39873b1ef695791e9900cdb33f8a761eef242859.jpg)

![4c97a540ea46dc795ff1362ca8695242bdd007534ba40ec44e78cb0d38b78731.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/4c97a540ea46dc795ff1362ca8695242bdd007534ba40ec44e78cb0d38b78731.jpg)

![4eb18071119c2f03a6d3fb35a37ae8e1043397e0115dc36c65858896ac54c3a6.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/4eb18071119c2f03a6d3fb35a37ae8e1043397e0115dc36c65858896ac54c3a6.jpg)

![77e418028f5da5cf47fb83ac9b57ed148a42f693dc5ac0044bbcc1b00cce2678.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/77e418028f5da5cf47fb83ac9b57ed148a42f693dc5ac0044bbcc1b00cce2678.jpg)

![921b88fd92ddfee8f56f59cf57ec6ac3be3b0ccc7e04a1feb202ff3758fc9551.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/921b88fd92ddfee8f56f59cf57ec6ac3be3b0ccc7e04a1feb202ff3758fc9551.jpg)

![a35fb0b19d25dab1bfcae79fb5a3805825c34f68e7c7f002c6d8edf9c0fc6a20.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/a35fb0b19d25dab1bfcae79fb5a3805825c34f68e7c7f002c6d8edf9c0fc6a20.jpg)

![d066123b6e203cc35838912b876746e40f0fa8a12896f4515d43ebbf290ddfdb.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/d066123b6e203cc35838912b876746e40f0fa8a12896f4515d43ebbf290ddfdb.jpg)

![f84da365171b0059311b35f13e79d548ba246d637dcba71627645141480689f0.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/images/f84da365171b0059311b35f13e79d548ba246d637dcba71627645141480689f0.jpg)

### Tables

![053f7f122f54daf7ccdde1a1ddedb63eccdc7e0ed21c988f7954b8b23f32d0cc.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/053f7f122f54daf7ccdde1a1ddedb63eccdc7e0ed21c988f7954b8b23f32d0cc.jpg)

![0a1f097b4cd35586f3d3625b688f14d2e62b37ba3a806b1d455977867d7625e9.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/0a1f097b4cd35586f3d3625b688f14d2e62b37ba3a806b1d455977867d7625e9.jpg)

![39fd8d31ce5207915131d30d898246fe23f886295f6a5a2d60d347241f8bcb3f.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/39fd8d31ce5207915131d30d898246fe23f886295f6a5a2d60d347241f8bcb3f.jpg)

![503e4439c614d9f511f11710039f3a76cdf7d988694ae5bb4137b9c78f679835.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/503e4439c614d9f511f11710039f3a76cdf7d988694ae5bb4137b9c78f679835.jpg)

![5836cbe6917b94c33bf0723b9afc1cf12920b4ef04f4bfd80dc5249fa9dbdc20.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/5836cbe6917b94c33bf0723b9afc1cf12920b4ef04f4bfd80dc5249fa9dbdc20.jpg)

![811bde08e96edacf88de1b5f87582911a814a20146e8c065572e2723e53cd97d.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/811bde08e96edacf88de1b5f87582911a814a20146e8c065572e2723e53cd97d.jpg)

![87ca939e1ee7f42adbe00cfc619b0b94d1dbf91fab72efe5a583dcb6ab2c2f3f.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/87ca939e1ee7f42adbe00cfc619b0b94d1dbf91fab72efe5a583dcb6ab2c2f3f.jpg)

![9e663b6d218a639b2b90b2eb58d1cd46d7fddb8c54bd9093e3a2760f9547187e.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/9e663b6d218a639b2b90b2eb58d1cd46d7fddb8c54bd9093e3a2760f9547187e.jpg)

![b71833839772e25dec58e165e5d5e5831c8df4e09e710670f350fd7180ca6ad4.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/b71833839772e25dec58e165e5d5e5831c8df4e09e710670f350fd7180ca6ad4.jpg)

![b7af4dd2df78a57706eb76e1af5af68cfed9f5a0e23136474d1b30f878b60c55.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/b7af4dd2df78a57706eb76e1af5af68cfed9f5a0e23136474d1b30f878b60c55.jpg)

![be186b1f6bdb3176c6793ff3ea368def125ce6e21b8ffd3c46368ed63840bb3e.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/be186b1f6bdb3176c6793ff3ea368def125ce6e21b8ffd3c46368ed63840bb3e.jpg)

![c963f08ca42d6d772a37e1ca8f4e929f759920a2081f95f09f1f22064a8e9548.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/c963f08ca42d6d772a37e1ca8f4e929f759920a2081f95f09f1f22064a8e9548.jpg)

![ceec9dd51ceeddfd7b34089eef983b14ef9669430d478656a391c753dbe95eb6.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/ceec9dd51ceeddfd7b34089eef983b14ef9669430d478656a391c753dbe95eb6.jpg)

![e91edff31502f1f3bfa08a87075c9131113606e35776d3090ad8726834a23227.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/e91edff31502f1f3bfa08a87075c9131113606e35776d3090ad8726834a23227.jpg)

![ec40c184c88fb6046ebf29313fe7d9bdab64ad9aecf876dc6c0765b5fac87c75.jpg](../icml_results/5161_Speedup%20Patch_%20Learning%20a%20Plug-and-Play%20Policy%20to%20Accelerate%20Embodied%20Manipulation/tables/ec40c184c88fb6046ebf29313fe7d9bdab64ad9aecf876dc6c0765b5fac87c75.jpg)

<a id="paper-5114"></a>

## Chain-of-Glimpse: Search-Guided Progressive Object-Grounded Reasoning for Video Understanding


### Images

![00eb3a6a3d05f10b814f0706d55d7d535236e7ced01d4da2684aa0cc09ca5a00.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/00eb3a6a3d05f10b814f0706d55d7d535236e7ced01d4da2684aa0cc09ca5a00.jpg)

![0d12bb60bda8ef59ccc39f792d684e76fb2b7732afd0361bf17476f52006e413.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/0d12bb60bda8ef59ccc39f792d684e76fb2b7732afd0361bf17476f52006e413.jpg)

![3d2e0767201f8dba0afa9bec75c816911ddd637859a37715d77473c25481fc41.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/3d2e0767201f8dba0afa9bec75c816911ddd637859a37715d77473c25481fc41.jpg)

![52fa25ceec91077214fbd6ba22e3bf9ee4877a0350018894c76d6c38b9bc7df1.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/52fa25ceec91077214fbd6ba22e3bf9ee4877a0350018894c76d6c38b9bc7df1.jpg)

![a024f1e3f160f905120e16e86ec0ea866f5e5272a63ae04ea0ae325bed45764f.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/a024f1e3f160f905120e16e86ec0ea866f5e5272a63ae04ea0ae325bed45764f.jpg)

![a35a7e936ebb47b795a77339565ae09042a2dd985f3463c2e26cb6188e0e148c.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/a35a7e936ebb47b795a77339565ae09042a2dd985f3463c2e26cb6188e0e148c.jpg)

![c221a53821b5e726e624cca64b48ef89187df2807d8704a4c6399e9fc5731145.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/images/c221a53821b5e726e624cca64b48ef89187df2807d8704a4c6399e9fc5731145.jpg)

### Tables

![19ebe28303a5c9976e71eaef53262a9e416b650e41e44f2fd6d574ba81c2ad73.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/19ebe28303a5c9976e71eaef53262a9e416b650e41e44f2fd6d574ba81c2ad73.jpg)

![21cb4eea6bf446baac062a6ccdb1ecf28b3abc1bf0cfdcd9aa606556124b0374.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/21cb4eea6bf446baac062a6ccdb1ecf28b3abc1bf0cfdcd9aa606556124b0374.jpg)

![27f0098c381a199b372070c2cccff5a78b94dc5f29407cfadfc06fd2d2134719.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/27f0098c381a199b372070c2cccff5a78b94dc5f29407cfadfc06fd2d2134719.jpg)

![5fd8adfb8f6d3b980431a962a5337ee207999784ac5c5188be11083aab967b08.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/5fd8adfb8f6d3b980431a962a5337ee207999784ac5c5188be11083aab967b08.jpg)

![913d480da6a59479dccf38407324f5f8960971b23c54989daa51ebee2c864e24.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/913d480da6a59479dccf38407324f5f8960971b23c54989daa51ebee2c864e24.jpg)

![ab33bc08248301a3e65fa795b23d64415fc1fc7adc4aef9e941d632a95e93905.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/ab33bc08248301a3e65fa795b23d64415fc1fc7adc4aef9e941d632a95e93905.jpg)

![b7aec666eaede3f1c9f31f3b58a16bf7755b72de46e1b9e55ff078e90b193ce3.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/b7aec666eaede3f1c9f31f3b58a16bf7755b72de46e1b9e55ff078e90b193ce3.jpg)

![c106ea78d2ef5083647c6548d0661936cc6cc2fdac23f6b554aa1474396cf46b.jpg](../icml_results/5162_Chain-of-Glimpse_%20Search-Guided%20Progressive%20Object-Grounded%20Reasoning%20for%20Video%20Understanding/tables/c106ea78d2ef5083647c6548d0661936cc6cc2fdac23f6b554aa1474396cf46b.jpg)

<a id="paper-5115"></a>

## Furina: Fragmented Uncertainty-Driven Refusal Instability Attack

### Images

![13ba2949439438d76e22733748a22c3f3b4f698da517f729e2cf086979cecd76.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/13ba2949439438d76e22733748a22c3f3b4f698da517f729e2cf086979cecd76.jpg)

![19db39eb5a6984901a020105caf99019b90d43d3aca5dd1640b7cda380b46671.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/19db39eb5a6984901a020105caf99019b90d43d3aca5dd1640b7cda380b46671.jpg)

![37e991c2a78ff077ef94ea01cb7dc988a87177dd31870a6258a3a8362b957c1d.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/37e991c2a78ff077ef94ea01cb7dc988a87177dd31870a6258a3a8362b957c1d.jpg)

![590bf80b8adcf37ac8af29a0ea02131f20f560ac899b4a60a1d9f21a518d42fc.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/590bf80b8adcf37ac8af29a0ea02131f20f560ac899b4a60a1d9f21a518d42fc.jpg)

![811798c2fe390dbbd8271e27d7bf49444ab8a59fbfc26563be1460932fa49e9d.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/811798c2fe390dbbd8271e27d7bf49444ab8a59fbfc26563be1460932fa49e9d.jpg)

![a01222e84970b3749c6f54f1a659e83b7d24a5d4438b79825eafd7b307edebfd.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/a01222e84970b3749c6f54f1a659e83b7d24a5d4438b79825eafd7b307edebfd.jpg)

![aa2813e33fc8b4a842e3e6eda568664f9072a479e5a42802424c3da7781fb49a.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/aa2813e33fc8b4a842e3e6eda568664f9072a479e5a42802424c3da7781fb49a.jpg)

![e6afeae01250477053b083b11bffc2665f0b9148bc9a7cf0e1ceb8bde31481fd.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/images/e6afeae01250477053b083b11bffc2665f0b9148bc9a7cf0e1ceb8bde31481fd.jpg)

### Tables

![04e20483395c0382f59a660d0b5234b81f3f0482012b24cbd052d6338355f8df.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/04e20483395c0382f59a660d0b5234b81f3f0482012b24cbd052d6338355f8df.jpg)

![079db83692a70432d784cd74a5533d9893fcfc618c76299452f6c82c0751afdc.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/079db83692a70432d784cd74a5533d9893fcfc618c76299452f6c82c0751afdc.jpg)

![0ab89662e1812f020f915551abf838653bd564d43e034a94ace292a3b65d44b6.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/0ab89662e1812f020f915551abf838653bd564d43e034a94ace292a3b65d44b6.jpg)

![5890cf8dd0d47fb0cdb5acfaf6ca3301cfec55512d0678e6103f43acee11922b.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/5890cf8dd0d47fb0cdb5acfaf6ca3301cfec55512d0678e6103f43acee11922b.jpg)

![a39b90a9a9cd81faac07cf3693b3ce143988b3aec82a038b3c04386820e65765.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/a39b90a9a9cd81faac07cf3693b3ce143988b3aec82a038b3c04386820e65765.jpg)

![b02793e86bd3da326a6b8b2d21cc0df4cac88785df483776344e1f5ef89b84c2.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/b02793e86bd3da326a6b8b2d21cc0df4cac88785df483776344e1f5ef89b84c2.jpg)

![bde83d168891eef800fecf6f3ac05f0b4f9ede1ad7f87fa76aa261be1d9b5fd1.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/bde83d168891eef800fecf6f3ac05f0b4f9ede1ad7f87fa76aa261be1d9b5fd1.jpg)

![c3e0f29b0ced5771a6828b353786e761ebb08efd003ecd05e3f1c2cbbeebbcfc.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/c3e0f29b0ced5771a6828b353786e761ebb08efd003ecd05e3f1c2cbbeebbcfc.jpg)

![f2e7b70efa1f70ab9875556683bc9cb174c988032d902b03cc3cae9c344d9133.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/f2e7b70efa1f70ab9875556683bc9cb174c988032d902b03cc3cae9c344d9133.jpg)

![f5e92f590223fe7360135f65e12328148598f953393da1b8c123ddfe540682cf.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/f5e92f590223fe7360135f65e12328148598f953393da1b8c123ddfe540682cf.jpg)

![feff04e64ff625ed10012e4a4b6d4fd8a1eb7b97e48a7682b3008fc054179336.jpg](../icml_results/5163_Furina_%20Fragmented%20Uncertainty-Driven%20Refusal%20Instability%20Attack/tables/feff04e64ff625ed10012e4a4b6d4fd8a1eb7b97e48a7682b3008fc054179336.jpg)
