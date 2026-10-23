# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 129 of 197

## 目录 (Table of Contents)

1. [Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions](#paper-4225)
2. [MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning](#paper-4226)
3. [Towards Realistic Lifelong Re-identification: Identity Recurrence with Changing Clothes](#paper-4227)
4. [A Tale of Two Graphs: Separating Knowledge Exploration from Outline Structure for Open-Ended Deep Research](#paper-4228)
5. [Learning What to Generate: A Reinforcement Learning-based Closed-Loop Augmentation Framework for Person Re-identification](#paper-4229)
6. [PromptPilot: Game-Theoretic Multi-Agent Prompt Optimization for Segment Anything](#paper-4230)
7. [OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale](#paper-4231)
8. [CyberGym-E2E: Scalable Real-World Benchmark for AI Agents’ End-to-End Cybersecurity Capabilities](#paper-4232)
9. [Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO](#paper-4233)
10. [SAM Audio: Segment Anything in Audio](#paper-4234)
11. [Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge](#paper-4235)
12. [RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling](#paper-4236)
13. [Exploring Nonlinear Pathway in Parameter Space for Machine Unlearning](#paper-4237)
14. [A Single Layer to Explain Them All: Understanding Massive Activations in Large Language Models](#paper-4238)
15. [VideoLoom: A Video Large Language Model for Joint Spatial-Temporal Understanding](#paper-4239)
16. [Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs](#paper-4240)
17. [Think Twice Before You Act: Protecting LLM Agents Against Tool Description Poisoning via Isolated Planning](#paper-4241)
18. [AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation](#paper-4242)
19. [Training Deep Spiking Neural Networks without Normalization](#paper-4243)
20. [AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](#paper-4244)
21. [Structured Multi-modal Graph Disentanglement for Psychiatric Diagnosis](#paper-4245)
22. [$τ$-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge](#paper-4246)
23. [CGRiC: Compositional Risk Certification for Structured LLM Outputs](#paper-4247)
24. [CalPro: Prior-Aware Evidential Conformal Prediction with Structure-Aware Sensitivity Bounds for Protein Structures](#paper-4248)
25. [Certificate-Guided Pruning for Stochastic Lipschitz Optimization](#paper-4249)
26. [What Reward Structure Enables Efficient Sparse-Reward RL? A Proof-of-Concept with Policy-Aware Matrix Completion](#paper-4250)
27. [Infinite-Precision Autoregressive Modeling for Vector Graphics and Layouts](#paper-4251)
28. [Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping](#paper-4252)
29. [Query-Based Asymmetric Modeling with Decoupled Input–Output Rates for Speech Restoration](#paper-4253)
30. [Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schrödinger Bridge Matching](#paper-4254)
31. [Parsimonious Learning-Augmented Online Metric Matching](#paper-4255)
32. [Overclocking Electrostatic Generative Models](#paper-4256)
33. [Responsible Text-to-Image Diffusion: Interpretable and Linearly Controllable Semantics for Fair and Safe Generation](#paper-4257)

---


<a id="paper-4225"></a>

## Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions

### Images

![00ad4e431a546a7d9d7425ace01f75e6f02741314ab881de147fe6431e5184ff.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/00ad4e431a546a7d9d7425ace01f75e6f02741314ab881de147fe6431e5184ff.jpg)

![02c87d42edab8d6cf1f787f3534137fed734a7f22fda5ae5d23ab5180941b388.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/02c87d42edab8d6cf1f787f3534137fed734a7f22fda5ae5d23ab5180941b388.jpg)

![181c4326f2230e7e3bc7b8d4bf6ec54b0729f3a5ddb4c15f67a9f172f58c5174.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/181c4326f2230e7e3bc7b8d4bf6ec54b0729f3a5ddb4c15f67a9f172f58c5174.jpg)

![1bef4a1e4ad98eb8b8719d5cf576b0fe41cb2af297bc344a42cb5ac6f3e533b1.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/1bef4a1e4ad98eb8b8719d5cf576b0fe41cb2af297bc344a42cb5ac6f3e533b1.jpg)

![2526c76f8019b4c94d8cdf967fdfd0b9c7eb965bce93af54d0751905688a439e.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/2526c76f8019b4c94d8cdf967fdfd0b9c7eb965bce93af54d0751905688a439e.jpg)

![28ac7b770662aa82baaf75cb4a11eb8cd0f959968107ef1689a29970ab57bded.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/28ac7b770662aa82baaf75cb4a11eb8cd0f959968107ef1689a29970ab57bded.jpg)

![6a3c84b2ddebd6ec861847fcc67be45aec6a9050639fd758712310689144afaf.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/6a3c84b2ddebd6ec861847fcc67be45aec6a9050639fd758712310689144afaf.jpg)

![74b5e94a486ba4c01b4c6928e2f6fbf0453c84becfd9eaad239d8612869a5ad5.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/74b5e94a486ba4c01b4c6928e2f6fbf0453c84becfd9eaad239d8612869a5ad5.jpg)

![8cc0d274e71766e8793c62844015feab2253ce04d53fd0f56b3a144dae77680c.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/8cc0d274e71766e8793c62844015feab2253ce04d53fd0f56b3a144dae77680c.jpg)

![c1963a0e9a4fdb87934f1d4e97be7f4286fd9a9cded7bdcb91db111557a0e087.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/c1963a0e9a4fdb87934f1d4e97be7f4286fd9a9cded7bdcb91db111557a0e087.jpg)

![f975fde6ca24ae06c838596bb43593b1f4f2a93a53ddcd7ecdcf90e2825656d4.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/images/f975fde6ca24ae06c838596bb43593b1f4f2a93a53ddcd7ecdcf90e2825656d4.jpg)

### Tables

![0268897f80dab8e467d67544c923d6857129c960d175b174f7b31da68ff89108.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/0268897f80dab8e467d67544c923d6857129c960d175b174f7b31da68ff89108.jpg)

![1f54b444112c6edf43ba6ae4380d8fc17e13527d18ce5f943c23d509451c5bd1.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/1f54b444112c6edf43ba6ae4380d8fc17e13527d18ce5f943c23d509451c5bd1.jpg)

![2853227bb4f581b5b61dffc65517baf92557702db655f9c06da25884d7001f70.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/2853227bb4f581b5b61dffc65517baf92557702db655f9c06da25884d7001f70.jpg)

![3f69a46d64eba2d05bd1bd59711e869a5d67e4121f2c196e2083414f2c2ca5a7.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/3f69a46d64eba2d05bd1bd59711e869a5d67e4121f2c196e2083414f2c2ca5a7.jpg)

![3fc12ac6d45feea69a284a5eeff21346a93b5bbc04e18a45477a88a5b7050e11.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/3fc12ac6d45feea69a284a5eeff21346a93b5bbc04e18a45477a88a5b7050e11.jpg)

![4856a0aa645f923cc52c53d01418ea19eea7f880bea3dc5b04ba3732c2f854c9.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/4856a0aa645f923cc52c53d01418ea19eea7f880bea3dc5b04ba3732c2f854c9.jpg)

![4b07ab1b64e7abd7f20640008f9d8672073df66d9ccc1de5826ede65dc95cc91.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/4b07ab1b64e7abd7f20640008f9d8672073df66d9ccc1de5826ede65dc95cc91.jpg)

![52ea21a2d75f8dd4ff9d55e4c208dd6bd10d7250401e28772eea0380a5f9e2c6.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/52ea21a2d75f8dd4ff9d55e4c208dd6bd10d7250401e28772eea0380a5f9e2c6.jpg)

![9ac3c77fef6ee2a21ae328b2ee1b9582b1a349b9cf023f81e1b00d1d3651cf31.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/9ac3c77fef6ee2a21ae328b2ee1b9582b1a349b9cf023f81e1b00d1d3651cf31.jpg)

![a44a3d47b0224ea41d2dba8a1640d5cb36ffcd9187a891523678257be0c7b352.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/a44a3d47b0224ea41d2dba8a1640d5cb36ffcd9187a891523678257be0c7b352.jpg)

![adb6398f5243b34af8b1aed0b5805e3f9dfad0acbb33b96ddcf5def22ede70e7.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/adb6398f5243b34af8b1aed0b5805e3f9dfad0acbb33b96ddcf5def22ede70e7.jpg)

![c010f0ccc134a7b48704df37599bf5aa934b0889390b3e73a508f0d0e5e09f18.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/c010f0ccc134a7b48704df37599bf5aa934b0889390b3e73a508f0d0e5e09f18.jpg)

![c7f1b994b0d7c993a42ba802a9f7f00fcf92c4cdf96387727aee76ef35e27f31.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/c7f1b994b0d7c993a42ba802a9f7f00fcf92c4cdf96387727aee76ef35e27f31.jpg)

![c80a54744a0a715e84320308898bc4c2d1a822cd2440726e5ae9bb38de18b84b.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/c80a54744a0a715e84320308898bc4c2d1a822cd2440726e5ae9bb38de18b84b.jpg)

![cb7b7bbb0a298d816d95c0e165e6d62025e74bfaec55be020af67001af1b37fd.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/cb7b7bbb0a298d816d95c0e165e6d62025e74bfaec55be020af67001af1b37fd.jpg)

![d6d04e848d636e15a97e8240e1ea15232923ee85ff7730e5446a6629122a3026.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/d6d04e848d636e15a97e8240e1ea15232923ee85ff7730e5446a6629122a3026.jpg)

![daf8ec8eb8d428a1b7666522f061a323d73d1d3b8d7dbd65e8fab57036ebe38f.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/daf8ec8eb8d428a1b7666522f061a323d73d1d3b8d7dbd65e8fab57036ebe38f.jpg)

![f045d856b05746ea50e5cce44f15f19ee70662530606b195a4bcef25d90bc75b.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/f045d856b05746ea50e5cce44f15f19ee70662530606b195a4bcef25d90bc75b.jpg)

![f352f543c4f8b951a0cb20b84ff72d9822ab6412179c97dbb1eb787a68525d2c.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/f352f543c4f8b951a0cb20b84ff72d9822ab6412179c97dbb1eb787a68525d2c.jpg)

![fe9e91ed53b26b8b8d15976f2ca0457e9116406a6c812d9166e00019e3236e62.jpg](../icml_results/4267_Tackling%20Fake%20Forgetting%20through%20Uncertainty%20Quantification/tables/fe9e91ed53b26b8b8d15976f2ca0457e9116406a6c812d9166e00019e3236e62.jpg)

## Understanding Performance Collapse in Layer-Pruned Large Language Models via Decision Representation Transitions


### Images

![0c18e56dc6a7e75d45c115a605886081bdda6d72c8d6a50f3ad56748ecd63d60.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/0c18e56dc6a7e75d45c115a605886081bdda6d72c8d6a50f3ad56748ecd63d60.jpg)

![110267cdf9f78e98e0d1749caadb13b886d16a18d583d1cffebf4e5168d6cb17.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/110267cdf9f78e98e0d1749caadb13b886d16a18d583d1cffebf4e5168d6cb17.jpg)

![224b8c1c9440bfcdbbddf4a48271a5a314000e2e62fd84781bcd8797eb3c399d.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/224b8c1c9440bfcdbbddf4a48271a5a314000e2e62fd84781bcd8797eb3c399d.jpg)

![4c9c336d0a57d6d14f2c15b52cbe1ae9bd5b4f439f7d0a65c30c08153daa90b5.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/4c9c336d0a57d6d14f2c15b52cbe1ae9bd5b4f439f7d0a65c30c08153daa90b5.jpg)

![52ea25cff77357337c15322aee8f42b44446742d9f2d853f6759d146ab0e884d.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/52ea25cff77357337c15322aee8f42b44446742d9f2d853f6759d146ab0e884d.jpg)

![6214537e962a6eb330c7e53d92b99c64862c02e86e610272e37c62e23c14cde9.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/6214537e962a6eb330c7e53d92b99c64862c02e86e610272e37c62e23c14cde9.jpg)

![64e3651393723b3ee5a8728dcb611dd3055192be39fef0bdde6b4b787d4eb7d4.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/64e3651393723b3ee5a8728dcb611dd3055192be39fef0bdde6b4b787d4eb7d4.jpg)

![6fd2339e8f0cd36301ccae9e4cc1e17da82e497999ca6a5acbad23999e45b5d2.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/6fd2339e8f0cd36301ccae9e4cc1e17da82e497999ca6a5acbad23999e45b5d2.jpg)

![71b2a25082757b1d7287d97b895a2cf82c924b67d70dc07e8215ce3606a4a29d.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/71b2a25082757b1d7287d97b895a2cf82c924b67d70dc07e8215ce3606a4a29d.jpg)

![784e030cc0654f1e57d579b56b4830eec734d0ab9a49f17110017ff6ee83fc89.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/784e030cc0654f1e57d579b56b4830eec734d0ab9a49f17110017ff6ee83fc89.jpg)

![871294c504ffe6adaa6c18cd78575c52a81a4defcfc4120e1463ae8ea652e3b4.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/871294c504ffe6adaa6c18cd78575c52a81a4defcfc4120e1463ae8ea652e3b4.jpg)

![877a6d73b4f7d553e7c3b1ff94fb6440b390627716a5d73d1022fc7bce3d1bbc.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/877a6d73b4f7d553e7c3b1ff94fb6440b390627716a5d73d1022fc7bce3d1bbc.jpg)

![a089c08c951b9dfd7a99988813054eac0da7bbc6f36e4971cdfc868a2bd7536c.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/a089c08c951b9dfd7a99988813054eac0da7bbc6f36e4971cdfc868a2bd7536c.jpg)

![a9f49cf33eb9243730553c6e45264ef8839be94db418adea2412486271c53649.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/a9f49cf33eb9243730553c6e45264ef8839be94db418adea2412486271c53649.jpg)

![b04d2907e132a97991676f1c7d294f6a7ef22ff97d2effd68adcb3b7bf5c620e.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/b04d2907e132a97991676f1c7d294f6a7ef22ff97d2effd68adcb3b7bf5c620e.jpg)

![c7f1e805492d250e37c863f7a4419ed4a0d809049e5af3343c05154760111dea.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/c7f1e805492d250e37c863f7a4419ed4a0d809049e5af3343c05154760111dea.jpg)

![cdf545ce445c19c6fe17c1e3442c5631815c7362f46534e9f6592908742ea7bd.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/cdf545ce445c19c6fe17c1e3442c5631815c7362f46534e9f6592908742ea7bd.jpg)

![d59f21a163efc3a3bf1e55c238e8665189213a261b5f9e482f81180b609bb78a.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/d59f21a163efc3a3bf1e55c238e8665189213a261b5f9e482f81180b609bb78a.jpg)

![d6e2bf13fe255200eb219fb185272c270e06b17fbabf3c539f969e1ad989d566.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/images/d6e2bf13fe255200eb219fb185272c270e06b17fbabf3c539f969e1ad989d566.jpg)

### Tables

![15bcfc3a10c4269effde6040470a0e83ae96d911bd7147028e0b160be8d351c4.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/15bcfc3a10c4269effde6040470a0e83ae96d911bd7147028e0b160be8d351c4.jpg)

![43dfe448d1b92a34554d06c8d2e5e18a254e682eb7ee1af5210a8b6de1dc5194.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/43dfe448d1b92a34554d06c8d2e5e18a254e682eb7ee1af5210a8b6de1dc5194.jpg)

![4c425624f6ac93d2dee32d7f011018eda401605bb4d8728faa90793cefff3ef8.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/4c425624f6ac93d2dee32d7f011018eda401605bb4d8728faa90793cefff3ef8.jpg)

![60e77bcbd4a604d236f411caf15c9f12bc1f8abb8794bbbebc1238773fe751a5.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/60e77bcbd4a604d236f411caf15c9f12bc1f8abb8794bbbebc1238773fe751a5.jpg)

![620fe9f8583dba143b91eeadfdf1d189916efa9c31f6d0589e34c653ace86252.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/620fe9f8583dba143b91eeadfdf1d189916efa9c31f6d0589e34c653ace86252.jpg)

![670692c5e86cc125b9030c318540201768c8789c27a11ed07e3dd08c10dca3e9.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/670692c5e86cc125b9030c318540201768c8789c27a11ed07e3dd08c10dca3e9.jpg)

![6f7d30eb8c15d3e70d573536510bfdbab440ece405502b32ba6429b232519ddf.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/6f7d30eb8c15d3e70d573536510bfdbab440ece405502b32ba6429b232519ddf.jpg)

![86e4d5fb5d2d0bc1ca29bfb755e546288fd13642e9c38a0116d95601b4b80adc.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/86e4d5fb5d2d0bc1ca29bfb755e546288fd13642e9c38a0116d95601b4b80adc.jpg)

![c2c1b1f07408aeea091e1364e4edabe49d86b0df91f21f85641eca322af8ae67.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/c2c1b1f07408aeea091e1364e4edabe49d86b0df91f21f85641eca322af8ae67.jpg)

![c801ad6c47987d19e671e40ee11b51cce067d805c4ea34b5d8ca12259e02aefa.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/c801ad6c47987d19e671e40ee11b51cce067d805c4ea34b5d8ca12259e02aefa.jpg)

![d21c2d27559fd89942587304213f5dafdee244232940a085589e679d8d92d752.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/d21c2d27559fd89942587304213f5dafdee244232940a085589e679d8d92d752.jpg)

![d25ed49f91718ad551b3d7d825a9dfe1b38248bbaf27f4c601e0601d4a1638be.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/d25ed49f91718ad551b3d7d825a9dfe1b38248bbaf27f4c601e0601d4a1638be.jpg)

![fb88242bba1e9997bab958c10effc9d29e24cb61fd8f0b0587a63b8f54a7c515.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/fb88242bba1e9997bab958c10effc9d29e24cb61fd8f0b0587a63b8f54a7c515.jpg)

![fea0a3086ce27654c0d3053a389854c284f54cdf4a2199b96cd4d3327fc05ec3.jpg](../icml_results/4268_Understanding%20Performance%20Collapse%20in%20Layer-Pruned%20Large%20Language%20Models%20via%20Decision%20Representation/tables/fea0a3086ce27654c0d3053a389854c284f54cdf4a2199b96cd4d3327fc05ec3.jpg)

<a id="paper-4226"></a>

## MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning


### Images

![13474ea1c6cbc6fc88e4bc9d2adf01f04aecfc2b68741497782203e6b08dfbc5.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/13474ea1c6cbc6fc88e4bc9d2adf01f04aecfc2b68741497782203e6b08dfbc5.jpg)

![41f125c42a08c9e04193098f6db4332d3c98b32bfece61fdaf1a84f5c8fdb737.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/41f125c42a08c9e04193098f6db4332d3c98b32bfece61fdaf1a84f5c8fdb737.jpg)

![5dfb03603fd9555b4e038771e894af626bd65ea44e1da7aa8f44ff6b25ee8e3b.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/5dfb03603fd9555b4e038771e894af626bd65ea44e1da7aa8f44ff6b25ee8e3b.jpg)

![688de06c84da3e945a965e5aca30410ae3be52d75fc38925a517889b66987f65.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/688de06c84da3e945a965e5aca30410ae3be52d75fc38925a517889b66987f65.jpg)

![9944ac1d8ed9c1631503346532d88343cf4f9ecaaa45d4bfb57375776296abc9.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/9944ac1d8ed9c1631503346532d88343cf4f9ecaaa45d4bfb57375776296abc9.jpg)

![ce4a3922b2824831283cae7c762c5a67b3737145e1b3aa038e3201acafb9c904.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/ce4a3922b2824831283cae7c762c5a67b3737145e1b3aa038e3201acafb9c904.jpg)

![dd151c3bc2a3be81e4643497bd665854e6563f807f5ddcf77f7a4ea4a49de8cb.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/dd151c3bc2a3be81e4643497bd665854e6563f807f5ddcf77f7a4ea4a49de8cb.jpg)

![ec11fd16b028774ce2ad6e1abffe11f96a38b4a8b316e0670462e353a539500e.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/images/ec11fd16b028774ce2ad6e1abffe11f96a38b4a8b316e0670462e353a539500e.jpg)

### Tables

![052aea75aba58f88eb8e5186a930287477981c4b15f798a37353bcd13ff1de90.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/052aea75aba58f88eb8e5186a930287477981c4b15f798a37353bcd13ff1de90.jpg)

![09386e17ebacafaf439344d60248118fce64cf6d5b64c7afddf4658c591ebfcb.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/09386e17ebacafaf439344d60248118fce64cf6d5b64c7afddf4658c591ebfcb.jpg)

![097228558f1e01482b8281f2b50e50554c533f2b880dd5c53d77efbc7df2ec78.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/097228558f1e01482b8281f2b50e50554c533f2b880dd5c53d77efbc7df2ec78.jpg)

![0999ccf719a38b30b88b0f97f0b35fa09b38c249f75d93edfa773b8f9273fb55.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/0999ccf719a38b30b88b0f97f0b35fa09b38c249f75d93edfa773b8f9273fb55.jpg)

![194123b82f1b128d2ccd87efb772cc311420f6008477a72a643d93507625f845.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/194123b82f1b128d2ccd87efb772cc311420f6008477a72a643d93507625f845.jpg)

![2139fb8a2af5744aad63a4155d145fd4bef4f1a42d741a26504cb192b00c2123.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/2139fb8a2af5744aad63a4155d145fd4bef4f1a42d741a26504cb192b00c2123.jpg)

![2d04e8e0ec55e770a8102506ccb6ff93e81e51757b49bdb824737cb60ae35f05.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/2d04e8e0ec55e770a8102506ccb6ff93e81e51757b49bdb824737cb60ae35f05.jpg)

![45d74482b89f7a0ad702851c0382799dab2ac157681123883498ba901e23b4b3.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/45d74482b89f7a0ad702851c0382799dab2ac157681123883498ba901e23b4b3.jpg)

![5f2f3e256c554b3a4cf3816a4e8880bdf7ebc95f2a8cacc00d614104addf8a37.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/5f2f3e256c554b3a4cf3816a4e8880bdf7ebc95f2a8cacc00d614104addf8a37.jpg)

![62d18d71fb0d38d067340de385b210d178965b198fc034a4b2ce9e82729749b9.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/62d18d71fb0d38d067340de385b210d178965b198fc034a4b2ce9e82729749b9.jpg)

![6813f798b6e5f571cc3ba95d85b93d0a0d47e2fc90656a9875ba7ea5761f5359.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/6813f798b6e5f571cc3ba95d85b93d0a0d47e2fc90656a9875ba7ea5761f5359.jpg)

![78e0fb8c2525d746c6c9762ebb207ac3e86e5220188c592dabdc9c189be862ad.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/78e0fb8c2525d746c6c9762ebb207ac3e86e5220188c592dabdc9c189be862ad.jpg)

![8879cb34e73262d15852e592edb9e39a9a342c78f531af7204567906c0726e99.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/8879cb34e73262d15852e592edb9e39a9a342c78f531af7204567906c0726e99.jpg)

![95a1f5617c4002ed0eaf22c88fdcb05acb6c6e1f3f8a0501261b79ce7fc1de88.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/95a1f5617c4002ed0eaf22c88fdcb05acb6c6e1f3f8a0501261b79ce7fc1de88.jpg)

![b3cddf1214b22f3d1890e409d18c67c1c70a374d26160b131a72c7fa323153e1.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/b3cddf1214b22f3d1890e409d18c67c1c70a374d26160b131a72c7fa323153e1.jpg)

![d41443f5a9e7ce009666304bd03ff299ee6bef72e849ea4b2bc48cee067f5018.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/d41443f5a9e7ce009666304bd03ff299ee6bef72e849ea4b2bc48cee067f5018.jpg)

![eaa13ab206f33d466fd8cb03aab298456af1f2172bebc37d4837946f60654a45.jpg](../icml_results/4269_MemOCR_%20Layout-Aware%20Visual%20Memory%20for%20Efficient%20Long-Horizon%20Reasoning/tables/eaa13ab206f33d466fd8cb03aab298456af1f2172bebc37d4837946f60654a45.jpg)

<a id="paper-4227"></a>

## Towards Realistic Lifelong Re-identification: Identity Recurrence with Changing Clothes


### Images

![552ca6c583a3ed84c5a3fd21f893de6a3759c9598f72a1561854ff95e041feed.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/images/552ca6c583a3ed84c5a3fd21f893de6a3759c9598f72a1561854ff95e041feed.jpg)

![5e0d649754ec6f93128ff69d8891a3f8ab1da16ad35d9948b328c857f631d767.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/images/5e0d649754ec6f93128ff69d8891a3f8ab1da16ad35d9948b328c857f631d767.jpg)

![8dbde3bf348fde1ee0149f4a18d68c47b3418e701b7a7b09842cc0f3a94d5a48.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/images/8dbde3bf348fde1ee0149f4a18d68c47b3418e701b7a7b09842cc0f3a94d5a48.jpg)

![979baeb8bc3481f427bb2a4eabc4c2132023d1f1c26dd2ca497950b9a5cdd4cd.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/images/979baeb8bc3481f427bb2a4eabc4c2132023d1f1c26dd2ca497950b9a5cdd4cd.jpg)

![bbf43ac66605bf0a5632baf4b56739215c34f1358743e96b5c7e68fab44ffb7e.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/images/bbf43ac66605bf0a5632baf4b56739215c34f1358743e96b5c7e68fab44ffb7e.jpg)

### Tables

![0742ba6fb4a39d1744d0424d7589461221968cbbf996eabcbec6fad4ecdbe175.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/tables/0742ba6fb4a39d1744d0424d7589461221968cbbf996eabcbec6fad4ecdbe175.jpg)

![2411664bb17002546ac61c311243abe831c7e7b780c02816ae39883f2df6a28a.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/tables/2411664bb17002546ac61c311243abe831c7e7b780c02816ae39883f2df6a28a.jpg)

![2f82b64cfa119ef5aa84710f3287692720be0f2737bb82da8b4a0b2e166c4036.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/tables/2f82b64cfa119ef5aa84710f3287692720be0f2737bb82da8b4a0b2e166c4036.jpg)

![6b3fda107806a384ae946503b37e1e1655d810933ded14b6438d90b0241d242a.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/tables/6b3fda107806a384ae946503b37e1e1655d810933ded14b6438d90b0241d242a.jpg)

![a0c1d53253e29d46ea081713c29e70cd98c434ba9c33c113f4992541dd034153.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/tables/a0c1d53253e29d46ea081713c29e70cd98c434ba9c33c113f4992541dd034153.jpg)

![f00962a45cd96d70af45999d067791ee54c1844f71b7469839f5b3d40b494d43.jpg](../icml_results/4270_Towards%20Realistic%20Lifelong%20Re-identification_%20Identity%20Recurrence%20with%20Changing%20Clothes/tables/f00962a45cd96d70af45999d067791ee54c1844f71b7469839f5b3d40b494d43.jpg)

<a id="paper-4228"></a>

## A Tale of Two Graphs: Separating Knowledge Exploration from Outline Structure for Open-Ended Deep Research


### Images

![13c6d10c0dfe9df1a4089dd208c490a34ce4f5a4698564f1e7983347b95bed0e.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/13c6d10c0dfe9df1a4089dd208c490a34ce4f5a4698564f1e7983347b95bed0e.jpg)

![1610e3858045676c0e1264423df1e0a9d1498d2208c058c9021b3d44fd762660.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/1610e3858045676c0e1264423df1e0a9d1498d2208c058c9021b3d44fd762660.jpg)

![7d0a7c64b352054e8cc429790431ce9622727b265fe372e596e842acfb525309.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/7d0a7c64b352054e8cc429790431ce9622727b265fe372e596e842acfb525309.jpg)

![89c59a70005a74cdef9346ba70b7c1e48ae4e81f58cd9b5b63ff55e2b1dc74c6.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/89c59a70005a74cdef9346ba70b7c1e48ae4e81f58cd9b5b63ff55e2b1dc74c6.jpg)

![b7e4a2dbfdb82e78c43789ec18ddd43e9d535abd157a022e4db3d311a54339ab.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/b7e4a2dbfdb82e78c43789ec18ddd43e9d535abd157a022e4db3d311a54339ab.jpg)

![ca633cb6998070db87a2fae2a7ecc75dfa456d9e2cc9fd6df5909ccaad0841e1.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/ca633cb6998070db87a2fae2a7ecc75dfa456d9e2cc9fd6df5909ccaad0841e1.jpg)

![fda85a7c77a296db572b09e357021eaf4c3fa2fefbb9867d63bd5562da41763b.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/images/fda85a7c77a296db572b09e357021eaf4c3fa2fefbb9867d63bd5562da41763b.jpg)

### Tables

![1172dd8c07fe7422961de8a4eb2120d4369c23bf9b473674f7884154e99e5178.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/1172dd8c07fe7422961de8a4eb2120d4369c23bf9b473674f7884154e99e5178.jpg)

![11d7bd9384b04d7eff21dcae6072f2e7fc6e5c60eda16ad778c9fd80a08dfc8b.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/11d7bd9384b04d7eff21dcae6072f2e7fc6e5c60eda16ad778c9fd80a08dfc8b.jpg)

![276a44b065e7dc57da4347ade99e4901454c4807eb47214cf3fbcc8156d8c091.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/276a44b065e7dc57da4347ade99e4901454c4807eb47214cf3fbcc8156d8c091.jpg)

![4f91018cfc05197f2fccf4aa5699fcda6e6ffa55bb16b395c2eaff6e41de3b92.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/4f91018cfc05197f2fccf4aa5699fcda6e6ffa55bb16b395c2eaff6e41de3b92.jpg)

![a5e3120c704be734965eaf432e61b6631c15f3cf3ff98d2e9ef451c3a0cf9adb.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/a5e3120c704be734965eaf432e61b6631c15f3cf3ff98d2e9ef451c3a0cf9adb.jpg)

![bcd9b71bde057461c83ead9a966cd8f6d747f808d84c84e3d040ed57cccad3ef.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/bcd9b71bde057461c83ead9a966cd8f6d747f808d84c84e3d040ed57cccad3ef.jpg)

![c164afd943cc58f2ab8a5e696a603a948878554a2d2f37707e91333ff7e93fcd.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/c164afd943cc58f2ab8a5e696a603a948878554a2d2f37707e91333ff7e93fcd.jpg)

![ee1fe24c6336457753b4ab498af337f69fd2f27833570b9482f800e73d6e89cb.jpg](../icml_results/4271_A%20Tale%20of%20Two%20Graphs_%20Separating%20Knowledge%20Exploration%20from%20Outline%20Structure%20for%20Open-Ended%20Deep%20Re/tables/ee1fe24c6336457753b4ab498af337f69fd2f27833570b9482f800e73d6e89cb.jpg)

<a id="paper-4229"></a>

## Learning What to Generate: A Reinforcement Learning-based Closed-Loop Augmentation Framework for Person Re-identification


### Images

![0c1ce30c081f0912d2a16abf48a1a0ceaa1333796c289cb2ad58b114eaf0b060.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/0c1ce30c081f0912d2a16abf48a1a0ceaa1333796c289cb2ad58b114eaf0b060.jpg)

![18256f67cddde2cac6b61f83ec17dc28ac36ed6bcf8b8b3fd3554b4188e1f7ce.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/18256f67cddde2cac6b61f83ec17dc28ac36ed6bcf8b8b3fd3554b4188e1f7ce.jpg)

![1c36f84cbd1cf8b6a29b53e7eaa8ad856bddc61b6076ac365d044c6e96e0eb48.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/1c36f84cbd1cf8b6a29b53e7eaa8ad856bddc61b6076ac365d044c6e96e0eb48.jpg)

![220456033461a6634075f382a38a7e06a38d1d2a98aa03e9f40c46146dcfdadb.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/220456033461a6634075f382a38a7e06a38d1d2a98aa03e9f40c46146dcfdadb.jpg)

![47f114e5ea8f7fba12ce14dc8a84580a69e9a4a9ead9d70aa3f3876a3aad47b3.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/47f114e5ea8f7fba12ce14dc8a84580a69e9a4a9ead9d70aa3f3876a3aad47b3.jpg)

![60c7ba6b0a33a6bf5fffa71100ed0ed64f99efa720741f28802ddcec9e910b47.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/60c7ba6b0a33a6bf5fffa71100ed0ed64f99efa720741f28802ddcec9e910b47.jpg)

![627acf959fbcb56d0d80bf8f53ddb71fb8f5aba0e15b83a58632a8efafff79d0.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/627acf959fbcb56d0d80bf8f53ddb71fb8f5aba0e15b83a58632a8efafff79d0.jpg)

![70adc05505e140475f4a34feafbfbc8ab2257cd20fe8c4ef0cae1160602f9923.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/70adc05505e140475f4a34feafbfbc8ab2257cd20fe8c4ef0cae1160602f9923.jpg)

![9cc09275378d5a7f366cd9ee3c61809b5eb43598165dc8f127cf5b42800b9ef7.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/9cc09275378d5a7f366cd9ee3c61809b5eb43598165dc8f127cf5b42800b9ef7.jpg)

![a98d1a61c6ecb7bfb602a3928581a42ca35532cb86749757939a77b28ac5fd9f.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/a98d1a61c6ecb7bfb602a3928581a42ca35532cb86749757939a77b28ac5fd9f.jpg)

![c5650bc45b32ece01af6d72edbbb2ebb0199e3d691f84191db07a7a1830d70bb.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/c5650bc45b32ece01af6d72edbbb2ebb0199e3d691f84191db07a7a1830d70bb.jpg)

![d65f82eb62f1b05cf1e9fe3cdebb95dd6383bd653ad9243b14bd38537973e092.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/d65f82eb62f1b05cf1e9fe3cdebb95dd6383bd653ad9243b14bd38537973e092.jpg)

![ddeefa47b923b1b0bc2fed377274ded045b43c0ad2c4bdc665c93f564af75eb1.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/ddeefa47b923b1b0bc2fed377274ded045b43c0ad2c4bdc665c93f564af75eb1.jpg)

![fba458e59f9e3e2e1fe58566563cc35996d62deef52f0683b3c0c61ff683b53f.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/images/fba458e59f9e3e2e1fe58566563cc35996d62deef52f0683b3c0c61ff683b53f.jpg)

### Tables

![3574139a912f81e04e86ece5d2f9a06e8fd4d765d64a1d4631be08a466753937.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/3574139a912f81e04e86ece5d2f9a06e8fd4d765d64a1d4631be08a466753937.jpg)

![49b72c3d2b7f76a56282f3c7f9882b900afcdc5b21eaedff58d557c6708a6794.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/49b72c3d2b7f76a56282f3c7f9882b900afcdc5b21eaedff58d557c6708a6794.jpg)

![58c22bf7dd6e0a3a57e5952fd563958eb60bbbbbcde02c9a951265cf2ba427e4.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/58c22bf7dd6e0a3a57e5952fd563958eb60bbbbbcde02c9a951265cf2ba427e4.jpg)

![59d89a5e5f6a3aa682625cb1d035d7de733549a5a891608b1447eaa654f97657.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/59d89a5e5f6a3aa682625cb1d035d7de733549a5a891608b1447eaa654f97657.jpg)

![7df9e874f7971b284dc7ff5c1a840d1c1c28ff1eb8fde96e0e063dda95e5455c.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/7df9e874f7971b284dc7ff5c1a840d1c1c28ff1eb8fde96e0e063dda95e5455c.jpg)

![90fe60e30f6f663f5a988e9f615bead2c3e57ef92c00cdfd2ca1ea1c28d19de1.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/90fe60e30f6f663f5a988e9f615bead2c3e57ef92c00cdfd2ca1ea1c28d19de1.jpg)

![97df966bef3a24cd9cafa7775c1a5e5fed6486dcc152bb1adbaffb5708692364.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/97df966bef3a24cd9cafa7775c1a5e5fed6486dcc152bb1adbaffb5708692364.jpg)

![c2106c660091ddf8b753970d35c0ac1087f3ffa74d77ac8f8d55d4fe0f5f720c.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/c2106c660091ddf8b753970d35c0ac1087f3ffa74d77ac8f8d55d4fe0f5f720c.jpg)

![f01ad393c226b4727fe622ce61ae71bf6856ccc012a86404b3cb4efe1c0b3844.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/f01ad393c226b4727fe622ce61ae71bf6856ccc012a86404b3cb4efe1c0b3844.jpg)

![f73493eecc1c496935987639a5c91cc74c702365e9d6775d407beb9ad851d4c1.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/f73493eecc1c496935987639a5c91cc74c702365e9d6775d407beb9ad851d4c1.jpg)

![fc8909ee196f3a07d36cf8502b6d375db052db1586c102f62c114d4748073db3.jpg](../icml_results/4272_Learning%20What%20to%20Generate_%20A%20Reinforcement%20Learning-based%20Closed-Loop%20Augmentation%20Framework%20for%20Per/tables/fc8909ee196f3a07d36cf8502b6d375db052db1586c102f62c114d4748073db3.jpg)

<a id="paper-4230"></a>

## PromptPilot: Game-Theoretic Multi-Agent Prompt Optimization for Segment Anything


### Images

![1a2bcecd9ca149f6b8bfbbbc4aadc40164a67a9badc9c914406effe86f2dd9ad.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/images/1a2bcecd9ca149f6b8bfbbbc4aadc40164a67a9badc9c914406effe86f2dd9ad.jpg)

![56c3d23b1df068d4b4d2dfa89e57f634597b35161a630f93d41ec549a7c34470.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/images/56c3d23b1df068d4b4d2dfa89e57f634597b35161a630f93d41ec549a7c34470.jpg)

![ae7b408bfd30534d46a02731b7bfb430b6fa0cecb69fe9d0cecb8453b40368ff.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/images/ae7b408bfd30534d46a02731b7bfb430b6fa0cecb69fe9d0cecb8453b40368ff.jpg)

### Tables

![56e407d22158377d0bb29381f2fbb6694363046a99234721bb1f2e12c346ca70.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/tables/56e407d22158377d0bb29381f2fbb6694363046a99234721bb1f2e12c346ca70.jpg)

![7d8cdddca0630a2e55afeaad52a589b6b3bd4ecf9163a8f14297b7499b417e09.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/tables/7d8cdddca0630a2e55afeaad52a589b6b3bd4ecf9163a8f14297b7499b417e09.jpg)

![a63ba5cdc2f8b3ad9d8be915b0b3fbd7862b1ca1e15c587e3ea36335b69b0a51.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/tables/a63ba5cdc2f8b3ad9d8be915b0b3fbd7862b1ca1e15c587e3ea36335b69b0a51.jpg)

![e8ead66b3b039fa7391784bcca968c632438b40eb4debcd4a36b4a70eca05269.jpg](../icml_results/4273_PromptPilot_%20Game-Theoretic%20Multi-Agent%20Prompt%20Optimization%20for%20Segment%20Anything/tables/e8ead66b3b039fa7391784bcca968c632438b40eb4debcd4a36b4a70eca05269.jpg)

<a id="paper-4231"></a>

## OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale


### Images

![0340842fedc08fdfe3fe49b063727ab2b5445431efd1444f02e73d241a4a0d06.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/0340842fedc08fdfe3fe49b063727ab2b5445431efd1444f02e73d241a4a0d06.jpg)

![203efc5eaff7496e4f28e1373cad4e638ba23155bdf386b9e18fc017e0160f88.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/203efc5eaff7496e4f28e1373cad4e638ba23155bdf386b9e18fc017e0160f88.jpg)

![25505ae7d50b3433cdb393598e93aa9080db8c6ad8afad833929d706dca05d11.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/25505ae7d50b3433cdb393598e93aa9080db8c6ad8afad833929d706dca05d11.jpg)

![41050af10e399a9e3f37eff826cfcf130d14b7b195f1d97895dbfc4a948cf466.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/41050af10e399a9e3f37eff826cfcf130d14b7b195f1d97895dbfc4a948cf466.jpg)

![5b3267382239d046385df620efdb688d78d189ac9e9e66c0e502ff419991e6b6.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/5b3267382239d046385df620efdb688d78d189ac9e9e66c0e502ff419991e6b6.jpg)

![79de5d333ccb2880689fd065dfd0accf51de66d57b91c0f86417d1f44869fde3.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/79de5d333ccb2880689fd065dfd0accf51de66d57b91c0f86417d1f44869fde3.jpg)

![afd27abeb34bd78af79fe3e368614fd1f9f7e37a16e99308bf098353373792e5.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/afd27abeb34bd78af79fe3e368614fd1f9f7e37a16e99308bf098353373792e5.jpg)

![c6c193e1c29ecc4c92fa1aa4cdccdbc3628127f9e940ab1e1b0716f652a3bf7d.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/images/c6c193e1c29ecc4c92fa1aa4cdccdbc3628127f9e940ab1e1b0716f652a3bf7d.jpg)

### Tables

![3b497b263b37b5f4a0d2ec781bb941109950789cc3306117b0bffe7fa7188b4a.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/tables/3b497b263b37b5f4a0d2ec781bb941109950789cc3306117b0bffe7fa7188b4a.jpg)

![3ce6ccc204d4941daee26ba9c2ef74eb05090cf764b0292cd819de063ba0a9f2.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/tables/3ce6ccc204d4941daee26ba9c2ef74eb05090cf764b0292cd819de063ba0a9f2.jpg)

![41db63217c2af509c638bb102334b1814d330bccae357f2382033d9feef24ce6.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/tables/41db63217c2af509c638bb102334b1814d330bccae357f2382033d9feef24ce6.jpg)

![a19ac2abc536310dfc127b9efaaac64b665bbdc3861349224b171f43f2536735.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/tables/a19ac2abc536310dfc127b9efaaac64b665bbdc3861349224b171f43f2536735.jpg)

![ecb72ea035356b4c62ff1fffdea6da65ca11adad0e7645c6af63cdebe37d1111.jpg](../icml_results/4274_OmniMoE_%20An%20Efficient%20MoE%20by%20Orchestrating%20Atomic%20Experts%20at%20Scale/tables/ecb72ea035356b4c62ff1fffdea6da65ca11adad0e7645c6af63cdebe37d1111.jpg)

<a id="paper-4232"></a>

## CyberGym-E2E: Scalable Real-World Benchmark for AI Agents’ End-to-End Cybersecurity Capabilities


### Images

![273e1c49d0ec6392657ece78cb070d8a7f2208c7054490a2c981028aa73b5e2f.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/images/273e1c49d0ec6392657ece78cb070d8a7f2208c7054490a2c981028aa73b5e2f.jpg)

![9f0b1b2fcdc3c8b08e73d1f36191cf6e9a88516ee74668109f7859aea9b5efef.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/images/9f0b1b2fcdc3c8b08e73d1f36191cf6e9a88516ee74668109f7859aea9b5efef.jpg)

![f50df8ed53e8c8492d76ceffb65fa1e43bdd52f2893cc2f440a5097a3ab739e9.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/images/f50df8ed53e8c8492d76ceffb65fa1e43bdd52f2893cc2f440a5097a3ab739e9.jpg)

### Tables

![5ac21b1b837d51bc7f163bde9a1f657ca066b692e9a404b91482b897d003ae1b.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/5ac21b1b837d51bc7f163bde9a1f657ca066b692e9a404b91482b897d003ae1b.jpg)

![6f77c5a5c0742a620e85845ca290d2d55f80817304f50fc8c57eee5301cc31f2.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/6f77c5a5c0742a620e85845ca290d2d55f80817304f50fc8c57eee5301cc31f2.jpg)

![7c88170f27d199373aa364997a2f8e3fb30bc746c9673c9e256eef073be8a6b2.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/7c88170f27d199373aa364997a2f8e3fb30bc746c9673c9e256eef073be8a6b2.jpg)

![85c6836c20de1e84d5fcb8a12104ff01aba26a868307f09af8847b5fc506a2b6.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/85c6836c20de1e84d5fcb8a12104ff01aba26a868307f09af8847b5fc506a2b6.jpg)

![b7466f949e3c8cbd6cb38fbaca021de6a0f8e73d00debf224fecf4bdf7db54f6.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/b7466f949e3c8cbd6cb38fbaca021de6a0f8e73d00debf224fecf4bdf7db54f6.jpg)

![d01923323dc1a8bb85002c6c549dbd973dfa3f19fa7c0122f8ad33ecb9c8a7f7.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/d01923323dc1a8bb85002c6c549dbd973dfa3f19fa7c0122f8ad33ecb9c8a7f7.jpg)

![d4bd20ccdecbbceedc593c091bff9d15e64c92dc5ed9e64cd6abd4a308d89e08.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/d4bd20ccdecbbceedc593c091bff9d15e64c92dc5ed9e64cd6abd4a308d89e08.jpg)

![e897b7ad824bc7c8262633df386f6c91ce4e162f678d6f46717d2da610c5a0be.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/e897b7ad824bc7c8262633df386f6c91ce4e162f678d6f46717d2da610c5a0be.jpg)

![f0b4f5d76ed7370cbc96ce181af2319b417b2514a08f43019c0c206d38f5af88.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/f0b4f5d76ed7370cbc96ce181af2319b417b2514a08f43019c0c206d38f5af88.jpg)

![fd41a7fe0b981a981aa8c0a7459793b4ea684ce6a7175e626568e74979fb2dfc.jpg](../icml_results/4275_CyberGym-E2E_%20Scalable%20Real-World%20Benchmark%20for%20AI%20Agents%E2%80%99%20End-to-End%20Cybersecurity%20Capabilities/tables/fd41a7fe0b981a981aa8c0a7459793b4ea684ce6a7175e626568e74979fb2dfc.jpg)

<a id="paper-4233"></a>

## Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO


### Images

![5866c56402f6679540a1451025ad63083ab18fd8554de88002683c9e49560a3d.jpg](../icml_results/4276_Understanding%20the%20Performance%20Gap%20in%20Preference%20Learning_%20A%20Dichotomy%20of%20RLHF%20and%20DPO/images/5866c56402f6679540a1451025ad63083ab18fd8554de88002683c9e49560a3d.jpg)

![6a882ef26a63373dda87f77949d99294f1933dd7f83b46a146db6e545fe3da94.jpg](../icml_results/4276_Understanding%20the%20Performance%20Gap%20in%20Preference%20Learning_%20A%20Dichotomy%20of%20RLHF%20and%20DPO/images/6a882ef26a63373dda87f77949d99294f1933dd7f83b46a146db6e545fe3da94.jpg)

![92a09c4673803605506d6b7fc46a762d32e475d89b7601f96c19b56ed5a0fca3.jpg](../icml_results/4276_Understanding%20the%20Performance%20Gap%20in%20Preference%20Learning_%20A%20Dichotomy%20of%20RLHF%20and%20DPO/images/92a09c4673803605506d6b7fc46a762d32e475d89b7601f96c19b56ed5a0fca3.jpg)

![aaf10e4530da7cf0d75aafe18dbba66cbeda51d36ef9363033baa0d3c954f053.jpg](../icml_results/4276_Understanding%20the%20Performance%20Gap%20in%20Preference%20Learning_%20A%20Dichotomy%20of%20RLHF%20and%20DPO/images/aaf10e4530da7cf0d75aafe18dbba66cbeda51d36ef9363033baa0d3c954f053.jpg)

### Tables

![6b4587c9e819fb8a1e29251ea1c44fbf1d36de0f553c1a3c5be802d71d0f7575.jpg](../icml_results/4276_Understanding%20the%20Performance%20Gap%20in%20Preference%20Learning_%20A%20Dichotomy%20of%20RLHF%20and%20DPO/tables/6b4587c9e819fb8a1e29251ea1c44fbf1d36de0f553c1a3c5be802d71d0f7575.jpg)

<a id="paper-4234"></a>

## SAM Audio: Segment Anything in Audio


### Images

![048857d0ae64e711888f0bec9aae8019050451644540e91eef15855822e3f9df.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/048857d0ae64e711888f0bec9aae8019050451644540e91eef15855822e3f9df.jpg)

![282191387950f8a7cfdbfc08ed484cb08af7fdb7991e3dec6240e34364a4e4c2.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/282191387950f8a7cfdbfc08ed484cb08af7fdb7991e3dec6240e34364a4e4c2.jpg)

![3beca19afc1e901fda969309c886e25e458d5dc5150b4d8d4e73ca81308ccf3e.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/3beca19afc1e901fda969309c886e25e458d5dc5150b4d8d4e73ca81308ccf3e.jpg)

![53759172a73d36a04f54d7865cd9b4f3ae637d10ba588a6ea66d653c84a07bff.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/53759172a73d36a04f54d7865cd9b4f3ae637d10ba588a6ea66d653c84a07bff.jpg)

![75e218466fe261ff59ceb95e42f2e006daa02130117f1ba67745c5683554b3d7.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/75e218466fe261ff59ceb95e42f2e006daa02130117f1ba67745c5683554b3d7.jpg)

![a8d2828665f9f20bae196d23bff5ff55408559e1b87db0d95378733fecc74801.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/a8d2828665f9f20bae196d23bff5ff55408559e1b87db0d95378733fecc74801.jpg)

![ba24749c5ba3f07b4f1530e6961eade650e90e80558aa701fa7a8acbead5bad4.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/ba24749c5ba3f07b4f1530e6961eade650e90e80558aa701fa7a8acbead5bad4.jpg)

![c242a7a72a6c170bdf3d590f5c591f2c0cdd3595c7ad29fb10a5b5dcb4b4d8a6.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/c242a7a72a6c170bdf3d590f5c591f2c0cdd3595c7ad29fb10a5b5dcb4b4d8a6.jpg)

![c5e1c673a243ebf7a7f040a2ad54d9b900266e7daffb74ffd8642bfd21aa9a5d.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/c5e1c673a243ebf7a7f040a2ad54d9b900266e7daffb74ffd8642bfd21aa9a5d.jpg)

![d0b66ae6c24fe7fde28d1ad57dcc8530637b20add3d19e98e5dd3185849b47d2.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/d0b66ae6c24fe7fde28d1ad57dcc8530637b20add3d19e98e5dd3185849b47d2.jpg)

![d67e9501765a3e4c6c9d8d9cfed641262ff8ed68ecbbdec0de528f65cb0e5dd7.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/d67e9501765a3e4c6c9d8d9cfed641262ff8ed68ecbbdec0de528f65cb0e5dd7.jpg)

![eb7aa798d39d720ef427c194755a4c0a74ee7adb358c2d0bbc014a59ecaa5200.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/images/eb7aa798d39d720ef427c194755a4c0a74ee7adb358c2d0bbc014a59ecaa5200.jpg)

### Tables

![04ed3d13ed3edf0c8743a1db8b1d8363e3cc1d1831f728f3f7233f440388c758.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/04ed3d13ed3edf0c8743a1db8b1d8363e3cc1d1831f728f3f7233f440388c758.jpg)

![07c561779b24d439f657f97353b07acc3f4303613e3923ed21d8e7fd66931f06.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/07c561779b24d439f657f97353b07acc3f4303613e3923ed21d8e7fd66931f06.jpg)

![12debbc120d9e9eb20626e657767d7760c6e8b3776f8399975ed5381f955c045.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/12debbc120d9e9eb20626e657767d7760c6e8b3776f8399975ed5381f955c045.jpg)

![1aef14cc4325569d1122e4b4e41c83de1c4d25875a43ede7ac41b290170ac053.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/1aef14cc4325569d1122e4b4e41c83de1c4d25875a43ede7ac41b290170ac053.jpg)

![1e596c42be06783e48ff62ffecde8e1f222038f7141acc9dd1059c5c8b525962.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/1e596c42be06783e48ff62ffecde8e1f222038f7141acc9dd1059c5c8b525962.jpg)

![22bad3a9e7ab38c60018d098665722c2041053981b38b19fe1894c54c978939e.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/22bad3a9e7ab38c60018d098665722c2041053981b38b19fe1894c54c978939e.jpg)

![7cf8de05287f1e58d3e0fdc97e98f9d0ab87d10759e917dcaf5f75ecebe62e22.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/7cf8de05287f1e58d3e0fdc97e98f9d0ab87d10759e917dcaf5f75ecebe62e22.jpg)

![800c2a860c2a47c9fce0539894b2f88ea81f478539d5283af957a645f26198d2.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/800c2a860c2a47c9fce0539894b2f88ea81f478539d5283af957a645f26198d2.jpg)

![8e590bbe1ef762355868c467b1f071c6b07541c63d93f02f98be841a7b87279b.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/8e590bbe1ef762355868c467b1f071c6b07541c63d93f02f98be841a7b87279b.jpg)

![a5bd60bfe0b0f9afa92984768d878389fe1dc62aeaaaaf732a3085ac57f12096.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/a5bd60bfe0b0f9afa92984768d878389fe1dc62aeaaaaf732a3085ac57f12096.jpg)

![ad3960e53807b6bacd3e2028e50c76b480d8b910844c1e414aa859d03da6c689.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/ad3960e53807b6bacd3e2028e50c76b480d8b910844c1e414aa859d03da6c689.jpg)

![b53fdbe971a7b838f66c88c2733dd1ad775225d812f691518c246ca4b4c6f84e.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/b53fdbe971a7b838f66c88c2733dd1ad775225d812f691518c246ca4b4c6f84e.jpg)

![b6787e920a271d7f2f88234b6539a78229bcbafe23aff5b2d2c169fa0773a2de.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/b6787e920a271d7f2f88234b6539a78229bcbafe23aff5b2d2c169fa0773a2de.jpg)

![c2535dd211b368b599938b6f1aa760586b24648dd781294dc79810a7050ec658.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/c2535dd211b368b599938b6f1aa760586b24648dd781294dc79810a7050ec658.jpg)

![c8e5e8765253ebbbcad1ac0797b41fd5c6f0b07e4c8e84567fa47e2a619ed0d5.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/c8e5e8765253ebbbcad1ac0797b41fd5c6f0b07e4c8e84567fa47e2a619ed0d5.jpg)

![ce992d26b8b590959a5962df632c39167c4eab2f09eeb2bbcd4a7dabf14752ff.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/ce992d26b8b590959a5962df632c39167c4eab2f09eeb2bbcd4a7dabf14752ff.jpg)

![d281deadd72259e029f8bb0b6e9ad2a9ca212cf3d2bfe666ab626dd6de8cd75b.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/d281deadd72259e029f8bb0b6e9ad2a9ca212cf3d2bfe666ab626dd6de8cd75b.jpg)

![db5616d1c44cd8df35722f3244acd22ee7cd86e8e6db217256ba22fc2a8ffd8c.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/db5616d1c44cd8df35722f3244acd22ee7cd86e8e6db217256ba22fc2a8ffd8c.jpg)

![e8ad789565f764d4997b478718ae0f722349f1ef5ac90895437ef803f2b9415b.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/e8ad789565f764d4997b478718ae0f722349f1ef5ac90895437ef803f2b9415b.jpg)

![f027a31258d83c76c03a001e31f9ae33be73d653a1e8d2d42adeb5cad1106836.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/f027a31258d83c76c03a001e31f9ae33be73d653a1e8d2d42adeb5cad1106836.jpg)

![f322975be73ad58ce785f86ba779b6d20132d8ceff4279f24d82ee0cbffecca7.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/f322975be73ad58ce785f86ba779b6d20132d8ceff4279f24d82ee0cbffecca7.jpg)

![f89bf67bb0422eddae76ad681745116911856431ed9b43ac3591414e8e983804.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/f89bf67bb0422eddae76ad681745116911856431ed9b43ac3591414e8e983804.jpg)

![f8b54627e00f02448b14cb6a7c148c5e07caf0b7ebc08c89864178f9d9f05310.jpg](../icml_results/4277_SAM%20Audio_%20Segment%20Anything%20in%20Audio/tables/f8b54627e00f02448b14cb6a7c148c5e07caf0b7ebc08c89864178f9d9f05310.jpg)

<a id="paper-4235"></a>

## Diffusion Language Model Parallel Decoding via Product-of-Experts Bridge


### Images

![17d9fcfa41b18d97f46646398d7dab96ce04dc4be28bc4dc4d0f21a737795b9f.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/images/17d9fcfa41b18d97f46646398d7dab96ce04dc4be28bc4dc4d0f21a737795b9f.jpg)

![2f7a5579037fc4668f70c1f16289602a351094f8721b01f4c786dd93e66c3880.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/images/2f7a5579037fc4668f70c1f16289602a351094f8721b01f4c786dd93e66c3880.jpg)

![37f9ed4724641872d3b6c07132c891ee3e2f6bcf9c212ee7a32ef723b53d0a68.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/images/37f9ed4724641872d3b6c07132c891ee3e2f6bcf9c212ee7a32ef723b53d0a68.jpg)

![6a0fd2d9fbaf7dc54ad6bd806ca0243daae42db149ce66d7745cf17881f69ba6.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/images/6a0fd2d9fbaf7dc54ad6bd806ca0243daae42db149ce66d7745cf17881f69ba6.jpg)

### Tables

![75ebd051c1d8357c60c66f06081013165b9dda6a55c23b7ec68873fdeef25e73.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/tables/75ebd051c1d8357c60c66f06081013165b9dda6a55c23b7ec68873fdeef25e73.jpg)

![92e31c23d9e42f5c884f94163db0efd9efa47acbd95be0169538ad38ee058a71.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/tables/92e31c23d9e42f5c884f94163db0efd9efa47acbd95be0169538ad38ee058a71.jpg)

![b2506eb3212807970b85f82f6f45f864c28938a897fed8755ea7dde74975aa92.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/tables/b2506eb3212807970b85f82f6f45f864c28938a897fed8755ea7dde74975aa92.jpg)

![c74d7cda7df9344f50fefb5ba65590ff71713118930eb2f24b8203444ef64fe7.jpg](../icml_results/4278_Diffusion%20Language%20Model%20Parallel%20Decoding%20via%20Product-of-Experts%20Bridge/tables/c74d7cda7df9344f50fefb5ba65590ff71713118930eb2f24b8203444ef64fe7.jpg)

<a id="paper-4236"></a>

## RealtimeTool: Parallel Decoding for Real-Time LLM Function Calling


### Images

![0b84cf29c2bf7a9252d9e4a76474131e76632bf7dc4ed7a9947fe05eb6f146e2.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/images/0b84cf29c2bf7a9252d9e4a76474131e76632bf7dc4ed7a9947fe05eb6f146e2.jpg)

![27bf89df7cbb700af7b7fe21f5ee30d623123afe7817e8b39736a315a5d2a85d.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/images/27bf89df7cbb700af7b7fe21f5ee30d623123afe7817e8b39736a315a5d2a85d.jpg)

![29d7a7e89f71ef7869d2d9144a9214ba13d69eef2e334d29c724e6a81bd393e4.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/images/29d7a7e89f71ef7869d2d9144a9214ba13d69eef2e334d29c724e6a81bd393e4.jpg)

![a53903006f66c362cd506850e50528ded5c59a1b61b955767da5b04d24e2e5f3.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/images/a53903006f66c362cd506850e50528ded5c59a1b61b955767da5b04d24e2e5f3.jpg)

![f51d78363aa7caaad2d273dc11b368558d9084a79bdd1970f47c30a8af8a6af2.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/images/f51d78363aa7caaad2d273dc11b368558d9084a79bdd1970f47c30a8af8a6af2.jpg)

### Tables

![00257bf9aa7316ed67e6d3f192f0af050578821f2a25078c4bce00f5c1e13a4e.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/00257bf9aa7316ed67e6d3f192f0af050578821f2a25078c4bce00f5c1e13a4e.jpg)

![0b47dae97db2da8fbbc7456310c34172c71de78a9667c62525e97400eb5d09ca.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/0b47dae97db2da8fbbc7456310c34172c71de78a9667c62525e97400eb5d09ca.jpg)

![18ca134c2520942bea59c27e9474beaacccea07046a1d427dba70372ceccd0db.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/18ca134c2520942bea59c27e9474beaacccea07046a1d427dba70372ceccd0db.jpg)

![1d1b72c8f3f43cfa32d53db9a24389198caf247061e46c084ffcd3c6c530d0f7.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/1d1b72c8f3f43cfa32d53db9a24389198caf247061e46c084ffcd3c6c530d0f7.jpg)

![21e34c08a3b5de364f4c4674a3ff50ea947432b5bdd065b117e2109c3d4606ce.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/21e34c08a3b5de364f4c4674a3ff50ea947432b5bdd065b117e2109c3d4606ce.jpg)

![4854ae30abccb42c7b06cb2eca598bf6028a4dbfbcfccb7bead5aba18c781569.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/4854ae30abccb42c7b06cb2eca598bf6028a4dbfbcfccb7bead5aba18c781569.jpg)

![4fa0e0f828bf8cd9cb503eb649d69a0b35df7b52118d33f7f6e1d43bb9661a09.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/4fa0e0f828bf8cd9cb503eb649d69a0b35df7b52118d33f7f6e1d43bb9661a09.jpg)

![5533a52749112913dd18e402b94933b7648fe1ee24787cd85f8c6f24b9945b74.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/5533a52749112913dd18e402b94933b7648fe1ee24787cd85f8c6f24b9945b74.jpg)

![5677a7bc6dba3f44d7f96f415af4f3e28b9bf9362d7c1bd56b04badfa8c079ef.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/5677a7bc6dba3f44d7f96f415af4f3e28b9bf9362d7c1bd56b04badfa8c079ef.jpg)

![69da3faef6b93b7842440dec4836b89de33a2934269dc6f86e3b98a8672e74a6.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/69da3faef6b93b7842440dec4836b89de33a2934269dc6f86e3b98a8672e74a6.jpg)

![7d5f08dd8fe4d97c5d7f6286b46783172551e6116a3c8afdc537b7c0f794b199.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/7d5f08dd8fe4d97c5d7f6286b46783172551e6116a3c8afdc537b7c0f794b199.jpg)

![81e9c7784215eb30a69405bd2ca142d004570001c72bac4f456b01cdfe2ca5a2.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/81e9c7784215eb30a69405bd2ca142d004570001c72bac4f456b01cdfe2ca5a2.jpg)

![885b5d4b909862e80076d2cdc4eaa4f67baeaea4ed7841174829f10ba3b24ed0.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/885b5d4b909862e80076d2cdc4eaa4f67baeaea4ed7841174829f10ba3b24ed0.jpg)

![8cc7b7f50a2a088924f6cb129ea8383103626f9c9b03dcc09e226d016c5a9149.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/8cc7b7f50a2a088924f6cb129ea8383103626f9c9b03dcc09e226d016c5a9149.jpg)

![ab0b0a4b3276dc8826b25230fe43ea5ef6ee23e342c235d793f28aad8d18f065.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/ab0b0a4b3276dc8826b25230fe43ea5ef6ee23e342c235d793f28aad8d18f065.jpg)

![c1d9940ba16c25df5ee0596e13684494aee38fdf84ebd38ee896ea3bd2ef774c.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/c1d9940ba16c25df5ee0596e13684494aee38fdf84ebd38ee896ea3bd2ef774c.jpg)

![c80d280fb7e2304c0f158bbd80a3cc8c6a26fa99f92af99f27b18083ead986b3.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/c80d280fb7e2304c0f158bbd80a3cc8c6a26fa99f92af99f27b18083ead986b3.jpg)

![cb1b7e89158585da704b9fd9e9298795b450405f2095b1b7186c49f3ee0f7c77.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/cb1b7e89158585da704b9fd9e9298795b450405f2095b1b7186c49f3ee0f7c77.jpg)

![cc82ae1b24f2372d2d987ecc6e4ace37126a1ba2a441bacfe834f301dd77646f.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/cc82ae1b24f2372d2d987ecc6e4ace37126a1ba2a441bacfe834f301dd77646f.jpg)

![cea31933d4404084801c329d4f4d59bc1e691444700d4b875726f62d3dcac772.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/cea31933d4404084801c329d4f4d59bc1e691444700d4b875726f62d3dcac772.jpg)

![d5b51e5e015b4496102c58f3e894f7133a72f5f760fbacf14b1b21ecd85555a2.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/d5b51e5e015b4496102c58f3e894f7133a72f5f760fbacf14b1b21ecd85555a2.jpg)

![ec354b12077dd53209c6f41aeccc8f8d4f50ee9c475db55d2af2bfcab7fd4ee9.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/ec354b12077dd53209c6f41aeccc8f8d4f50ee9c475db55d2af2bfcab7fd4ee9.jpg)

![f5ea59732bc5da3459d0c16b5aa6d020b17e9d2464df2636569c287f38c78908.jpg](../icml_results/4279_RealtimeTool_%20Parallel%20Decoding%20for%20Real-Time%20LLM%20Function%20Calling/tables/f5ea59732bc5da3459d0c16b5aa6d020b17e9d2464df2636569c287f38c78908.jpg)

<a id="paper-4237"></a>

## Exploring Nonlinear Pathway in Parameter Space for Machine Unlearning


### Images

![06493e0009711bef2d41d2d5a5980e41c4b7ffb4a353dbdbae498c5d80ecac93.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/06493e0009711bef2d41d2d5a5980e41c4b7ffb4a353dbdbae498c5d80ecac93.jpg)

![0fd842f97a4b73691c79f6d012db7fa01afcc07efe46b26659aef133b5d5fee3.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/0fd842f97a4b73691c79f6d012db7fa01afcc07efe46b26659aef133b5d5fee3.jpg)

![3c02bc9f44a7963d7203fc71b58cf16624c2efef66b17beba93e338d57d6e78e.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/3c02bc9f44a7963d7203fc71b58cf16624c2efef66b17beba93e338d57d6e78e.jpg)

![48a6779c69118ad890c0071d42a89858b0c291c53c5a8578a86cfa8dbd08e918.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/48a6779c69118ad890c0071d42a89858b0c291c53c5a8578a86cfa8dbd08e918.jpg)

![75028b4fce7470b043cf665457d8c205e44109af2c6d8166c5b956a9b7071b57.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/75028b4fce7470b043cf665457d8c205e44109af2c6d8166c5b956a9b7071b57.jpg)

![7941600302a85e356c9448e97f3b46844d916b57ec467221f2f0372606efbfef.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/7941600302a85e356c9448e97f3b46844d916b57ec467221f2f0372606efbfef.jpg)

![8917e9aeb4dd95302449c812116a39097785536822559b60447628c11ecdb530.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/8917e9aeb4dd95302449c812116a39097785536822559b60447628c11ecdb530.jpg)

![8b4e72f3b5b6b494d6a4be02f00344ab25f8be4442edd6fb4a295c90e162207b.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/8b4e72f3b5b6b494d6a4be02f00344ab25f8be4442edd6fb4a295c90e162207b.jpg)

![975ddf112f7159f3bc7abbc25c4d62e96ba50aa25996efac36e5a5fb7cbac856.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/975ddf112f7159f3bc7abbc25c4d62e96ba50aa25996efac36e5a5fb7cbac856.jpg)

![99004fbd2a0c37e955934ec1b0c5bfff8a30da6556f4e6a3f5f8545bc503df0c.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/99004fbd2a0c37e955934ec1b0c5bfff8a30da6556f4e6a3f5f8545bc503df0c.jpg)

![cfc2b2a1b7e4da39e93517fb3037beed55110e7320d0d9a64a166d1afff29c4b.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/cfc2b2a1b7e4da39e93517fb3037beed55110e7320d0d9a64a166d1afff29c4b.jpg)

![d31b4ae83b9c49d19cf0068d848110b44696ad610854b566a16e2dc077b81fea.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/d31b4ae83b9c49d19cf0068d848110b44696ad610854b566a16e2dc077b81fea.jpg)

![dc50244fec31b97fa97d387cf4181b36ab16e5ed92bdbd2f3ec12d8bff604013.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/dc50244fec31b97fa97d387cf4181b36ab16e5ed92bdbd2f3ec12d8bff604013.jpg)

![dcf5340ecd761521610e54af968f24e4c6728e18dd8d5716977a0ee2b0abac8d.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/dcf5340ecd761521610e54af968f24e4c6728e18dd8d5716977a0ee2b0abac8d.jpg)

![e59eca8d1791bde0dd291040d1bd7350611f1b34de324b3b48d1348b3c3ddb18.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/images/e59eca8d1791bde0dd291040d1bd7350611f1b34de324b3b48d1348b3c3ddb18.jpg)

### Tables

![037c8780b737c6809c82be0487d44720b1ead6f3157c6af8ca11d16a9e9f9911.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/037c8780b737c6809c82be0487d44720b1ead6f3157c6af8ca11d16a9e9f9911.jpg)

![1ac42900b849eb175ff1e15912a4550d71b036e44f90c68f3e82e4e947540b32.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/1ac42900b849eb175ff1e15912a4550d71b036e44f90c68f3e82e4e947540b32.jpg)

![20aeb883fd26bda4ee27ac4bb2f9a242e87db4fde27cce44898d49d9a716c524.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/20aeb883fd26bda4ee27ac4bb2f9a242e87db4fde27cce44898d49d9a716c524.jpg)

![212852034fa67820d725c68191235bdb85b08442e10d75e85d7beee87095f541.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/212852034fa67820d725c68191235bdb85b08442e10d75e85d7beee87095f541.jpg)

![2d3388d21e10c990f8d6956c4e00de02f3c0c7a8ce53e07e5e55280877ac5874.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/2d3388d21e10c990f8d6956c4e00de02f3c0c7a8ce53e07e5e55280877ac5874.jpg)

![42e632e4edb9fc0b5d8980ce04503da3bf42f9739fc2b20f2282c90e1ca7cba9.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/42e632e4edb9fc0b5d8980ce04503da3bf42f9739fc2b20f2282c90e1ca7cba9.jpg)

![5d0976f3ef907b840e25707492fb8a799e062e716de22ff7a7876bc6bd6b9e6b.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/5d0976f3ef907b840e25707492fb8a799e062e716de22ff7a7876bc6bd6b9e6b.jpg)

![8bc0a56ffbf4f191506f3d46fba4ed3cfac2435af6070d3321bcf56f37308d99.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/8bc0a56ffbf4f191506f3d46fba4ed3cfac2435af6070d3321bcf56f37308d99.jpg)

![ce5cad0f56e421fa22322f864c47f33c8b22bdb658de7bfd8906431a19379d85.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/ce5cad0f56e421fa22322f864c47f33c8b22bdb658de7bfd8906431a19379d85.jpg)

![d73d20edf86155c1a76ee440d9fefb5d7575ecc410b5095223a6fd83a2cf2ade.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/d73d20edf86155c1a76ee440d9fefb5d7575ecc410b5095223a6fd83a2cf2ade.jpg)

![e46cac7be8f80c3a5c8caa98853705b6f848b541bb7c9b69cf7555b0b3f6ded7.jpg](../icml_results/4280_Exploring%20Nonlinear%20Pathway%20in%20Parameter%20Space%20for%20Machine%20Unlearning/tables/e46cac7be8f80c3a5c8caa98853705b6f848b541bb7c9b69cf7555b0b3f6ded7.jpg)

<a id="paper-4238"></a>

## A Single Layer to Explain Them All: Understanding Massive Activations in Large Language Models


### Images

![03dafb11937df6ef7214bc8595f3a3a712fa7e67253a4616723d07e6649e9d6f.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/03dafb11937df6ef7214bc8595f3a3a712fa7e67253a4616723d07e6649e9d6f.jpg)

![1cc49e7c5bc91dab7481801a9db6fa0a0ae6db9538110820ae7f8095d1702166.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/1cc49e7c5bc91dab7481801a9db6fa0a0ae6db9538110820ae7f8095d1702166.jpg)

![2903c9a5d8a22a430c6047cf6cd250365fc8afca04df8fcfc814a95123558dd4.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/2903c9a5d8a22a430c6047cf6cd250365fc8afca04df8fcfc814a95123558dd4.jpg)

![538f0577bc7fcbb556a7178ae0049847467090c62d108d3b9da0237edce0039e.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/538f0577bc7fcbb556a7178ae0049847467090c62d108d3b9da0237edce0039e.jpg)

![5f056745ee8b6a1600f225f58ae0b9b6e730ffe03b6a3137e70d8fb88d2d58b3.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/5f056745ee8b6a1600f225f58ae0b9b6e730ffe03b6a3137e70d8fb88d2d58b3.jpg)

![61761a6e6c83d1f37cdec82ce9aad1703c62b1aa8f50de30d344dc6541635efc.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/61761a6e6c83d1f37cdec82ce9aad1703c62b1aa8f50de30d344dc6541635efc.jpg)

![7410b42002ad1143efd7b2d22366785e49b9d1f2815f13b994ae28d82028499a.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/7410b42002ad1143efd7b2d22366785e49b9d1f2815f13b994ae28d82028499a.jpg)

![7afa1770871751313f65ab1fbdf29bfd7de311724afd0b9cbe2b2df857248548.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/7afa1770871751313f65ab1fbdf29bfd7de311724afd0b9cbe2b2df857248548.jpg)

![81ddf5fcedf893e21cad2fea9ff47e5ab69221d8b2392a23e074ba68d0face67.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/81ddf5fcedf893e21cad2fea9ff47e5ab69221d8b2392a23e074ba68d0face67.jpg)

![837afaf41c42d7d0787f13810e77b52bea6963e9d5b8d04807c414a072508d6b.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/837afaf41c42d7d0787f13810e77b52bea6963e9d5b8d04807c414a072508d6b.jpg)

![8b9822f986312c77c5320ed192be3e106e0b6030ffff387a26aa157387c036a0.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/8b9822f986312c77c5320ed192be3e106e0b6030ffff387a26aa157387c036a0.jpg)

![ad33c82c63cccfc2d62557a560999acbbd2ca0ab786218a058d9eff49183b039.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/ad33c82c63cccfc2d62557a560999acbbd2ca0ab786218a058d9eff49183b039.jpg)

![addaea09a66228a970f0d1087fd7bca8db2822b110751688f24dfd050f5cb7b0.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/addaea09a66228a970f0d1087fd7bca8db2822b110751688f24dfd050f5cb7b0.jpg)

![b5577085962f38a12f9c8a62bef65de1d819d07ec813bbb28c5142714ac41c3b.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/b5577085962f38a12f9c8a62bef65de1d819d07ec813bbb28c5142714ac41c3b.jpg)

![c27adf76d0a54fdebf47d3d1e903651b846d2a1b64455dc1811ad99b4845faa4.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/c27adf76d0a54fdebf47d3d1e903651b846d2a1b64455dc1811ad99b4845faa4.jpg)

![e7da12f85672fbefbb155b93872516c40ecaf5d499b392318ef818912b9269f5.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/e7da12f85672fbefbb155b93872516c40ecaf5d499b392318ef818912b9269f5.jpg)

![ea073e1f3730003b8b0c7103821c019468671efba0f3e098b5593cebd9b369db.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/ea073e1f3730003b8b0c7103821c019468671efba0f3e098b5593cebd9b369db.jpg)

![f0036bff3c73ca6ad16cf0e64c2b3a9a6fbeddb03432a67c19afefb89c42298f.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/f0036bff3c73ca6ad16cf0e64c2b3a9a6fbeddb03432a67c19afefb89c42298f.jpg)

![f6cb69710cc7633a9d256be51dd15a1f4db7263f29768d2d2e408fa27c010001.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/f6cb69710cc7633a9d256be51dd15a1f4db7263f29768d2d2e408fa27c010001.jpg)

![f89362e6bd95de55ba110774903342362bec7a8685bbe28bbfacf1d4e7b91f35.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/images/f89362e6bd95de55ba110774903342362bec7a8685bbe28bbfacf1d4e7b91f35.jpg)

### Tables

![74f7d5e307046393db86b0c13b570d28b5a5db95b647630740522cd5ac3a613f.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/74f7d5e307046393db86b0c13b570d28b5a5db95b647630740522cd5ac3a613f.jpg)

![80eb97c1619bd236dc6d515ccef1d9b619572e79e7a11dc78c9f60f0339219d7.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/80eb97c1619bd236dc6d515ccef1d9b619572e79e7a11dc78c9f60f0339219d7.jpg)

![84c1130b5f45fb3ef948937ba1fd8d60f0776c9b9d9751fa6645925029420112.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/84c1130b5f45fb3ef948937ba1fd8d60f0776c9b9d9751fa6645925029420112.jpg)

![b0023408c4ccb580716adff2e99015d065be4a45a98c8d04b282c4a345409a85.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/b0023408c4ccb580716adff2e99015d065be4a45a98c8d04b282c4a345409a85.jpg)

![be05458a0bd45484b511833f199834b7595d88fe8aa7959ae46a83035981dd3a.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/be05458a0bd45484b511833f199834b7595d88fe8aa7959ae46a83035981dd3a.jpg)

![c28ca1dd3aaa5f2764f1e8e99003307f6fd06e88e88055d0aeaa32bbbd60d47a.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/c28ca1dd3aaa5f2764f1e8e99003307f6fd06e88e88055d0aeaa32bbbd60d47a.jpg)

![d39b77a6d71c1bff53a313317c95f07b7cae2c064d43bd17955ee93409d1d466.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/d39b77a6d71c1bff53a313317c95f07b7cae2c064d43bd17955ee93409d1d466.jpg)

![fc728a70430e3d53e3d6c1a58d2f03525e6e691d01cd4709e8cd45a3257f19c4.jpg](../icml_results/4281_A%20Single%20Layer%20to%20Explain%20Them%20All_%20Understanding%20Massive%20Activations%20in%20Large%20Language%20Models/tables/fc728a70430e3d53e3d6c1a58d2f03525e6e691d01cd4709e8cd45a3257f19c4.jpg)

<a id="paper-4239"></a>

## VideoLoom: A Video Large Language Model for Joint Spatial-Temporal Understanding


### Images

![0abe29b422c08f033193976b15381dcc65848e5d1ec738296a477cdd9555599d.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/0abe29b422c08f033193976b15381dcc65848e5d1ec738296a477cdd9555599d.jpg)

![2532f6c098fde12994cda7625416feee7027cf95b1b037f7dd9e17961b162714.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/2532f6c098fde12994cda7625416feee7027cf95b1b037f7dd9e17961b162714.jpg)

![3108a2aa26ecd059eb3de0f93a29a482948d206c9b62b09b1abc1dfe5750e5c0.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/3108a2aa26ecd059eb3de0f93a29a482948d206c9b62b09b1abc1dfe5750e5c0.jpg)

![370692b1c6c92400e146e9603d9d2efbf4f9c4ac9eb3b164e6df93267f9656ff.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/370692b1c6c92400e146e9603d9d2efbf4f9c4ac9eb3b164e6df93267f9656ff.jpg)

![41a5fee826b3a0a76134e1978b5c82e5aac770f858c884763aa4164f52160c4a.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/41a5fee826b3a0a76134e1978b5c82e5aac770f858c884763aa4164f52160c4a.jpg)

![4bd8233b06469405d9f075e3293857b369677c6f6843efb81648c23ea9db4199.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/4bd8233b06469405d9f075e3293857b369677c6f6843efb81648c23ea9db4199.jpg)

![59df70179340d4ef90f76ea742dd8088825400e7cf63a92b8999de3112fbc571.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/59df70179340d4ef90f76ea742dd8088825400e7cf63a92b8999de3112fbc571.jpg)

![68dd554b72deaa0e97eec6e109375aeead79fa2d36bde5251b94fd271d373a88.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/68dd554b72deaa0e97eec6e109375aeead79fa2d36bde5251b94fd271d373a88.jpg)

![743b4fd060ae9f7f6909c922cb226a1af0058641193338126d2682c70ac500c9.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/743b4fd060ae9f7f6909c922cb226a1af0058641193338126d2682c70ac500c9.jpg)

![81e13f3b6116167e119d6483622b306d59047f7162b9796e9950491b05537e67.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/81e13f3b6116167e119d6483622b306d59047f7162b9796e9950491b05537e67.jpg)

![8e5db9d7b25f35f68511d52b10367f554a2d8bed514ecae7f245809e7ffcb24d.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/8e5db9d7b25f35f68511d52b10367f554a2d8bed514ecae7f245809e7ffcb24d.jpg)

![9e099bcd781c8fd86ec3efd5ecf5910ebcba055408cc1e9b40ac8d967326205f.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/9e099bcd781c8fd86ec3efd5ecf5910ebcba055408cc1e9b40ac8d967326205f.jpg)

![9eea24ff3abc358b8d4156bfd9dd1935a5078209ce851dfa96822284eb5b24f1.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/9eea24ff3abc358b8d4156bfd9dd1935a5078209ce851dfa96822284eb5b24f1.jpg)

![a12c589a119ca576a6c11828235b32812d7c888a6222e9ba7fd3956951b070ae.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/a12c589a119ca576a6c11828235b32812d7c888a6222e9ba7fd3956951b070ae.jpg)

![a2509d2b67481cbb2f67625202010f7640f1b39a461086990d173b4438733e6c.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/a2509d2b67481cbb2f67625202010f7640f1b39a461086990d173b4438733e6c.jpg)

![ac23c6c1afaa22d1d946fbdbfb3949145fe790c00f34d0e8e029afac4230dbd7.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/ac23c6c1afaa22d1d946fbdbfb3949145fe790c00f34d0e8e029afac4230dbd7.jpg)

![ac541b0f279dcb4e6bb702818348d85d49b1a8ea197c2aba75a44c4a1151df2d.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/ac541b0f279dcb4e6bb702818348d85d49b1a8ea197c2aba75a44c4a1151df2d.jpg)

![e9cf70b264e02a85c8be8f34357a3d88652acbf93877ad127a9558d611b80de9.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/e9cf70b264e02a85c8be8f34357a3d88652acbf93877ad127a9558d611b80de9.jpg)

![f137bcc7b53ebe4c1826b12a64248d04ef6c630d3fb73f5e7cff087f217fce93.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/f137bcc7b53ebe4c1826b12a64248d04ef6c630d3fb73f5e7cff087f217fce93.jpg)

![f1385764d841ca17263d26eb50818ad7994b664d12f3f5e1556e5f92acd10e52.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/images/f1385764d841ca17263d26eb50818ad7994b664d12f3f5e1556e5f92acd10e52.jpg)

### Tables

![0d6ba201fce981943b1e447a2b16ba4102a107d71823d7f0f019689ae57dd5c5.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/0d6ba201fce981943b1e447a2b16ba4102a107d71823d7f0f019689ae57dd5c5.jpg)

![1631e9cde980a39a1b838aea3dd2132ccb1d9fddc88c463fe888211a2f05f370.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/1631e9cde980a39a1b838aea3dd2132ccb1d9fddc88c463fe888211a2f05f370.jpg)

![228343d297bc4e8c2fb6b269aeee9e682be1f4063eb074cc54d0a2d26a4caf35.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/228343d297bc4e8c2fb6b269aeee9e682be1f4063eb074cc54d0a2d26a4caf35.jpg)

![276fc2f3ba5c3afd3fcf2bdcaf167fac73ac2790d840100dff77c6675777c904.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/276fc2f3ba5c3afd3fcf2bdcaf167fac73ac2790d840100dff77c6675777c904.jpg)

![359c84dd1f164c2fc3c61296f3f2e5c76de4a3998ba731faee7b8729917e2c4a.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/359c84dd1f164c2fc3c61296f3f2e5c76de4a3998ba731faee7b8729917e2c4a.jpg)

![438818e90a882720b275f731093472774d2aa607d0830b2c667d1cda1c6720ef.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/438818e90a882720b275f731093472774d2aa607d0830b2c667d1cda1c6720ef.jpg)

![4d05394f7686b4bb10672d8e4d56e0aebbba2ba107310d1534752e5b77afd7d1.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/4d05394f7686b4bb10672d8e4d56e0aebbba2ba107310d1534752e5b77afd7d1.jpg)

![4fc95740610ea3d37fe4c5dbcb42bfc86c9192dec95bd926a78d447371dce1f6.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/4fc95740610ea3d37fe4c5dbcb42bfc86c9192dec95bd926a78d447371dce1f6.jpg)

![53c606ca9407cc42beb2fa34ee57789041b0293c10b28d72a7d540c819b35da7.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/53c606ca9407cc42beb2fa34ee57789041b0293c10b28d72a7d540c819b35da7.jpg)

![6ba5af8ab049436521b15b190e331dfd077ff766cec1621cfdd5958000d0b11b.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/6ba5af8ab049436521b15b190e331dfd077ff766cec1621cfdd5958000d0b11b.jpg)

![73623c2d8e5b2cc302a938fa9e9fe48df734d1c374209afeb49b4824a264021a.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/73623c2d8e5b2cc302a938fa9e9fe48df734d1c374209afeb49b4824a264021a.jpg)

![99d46231870456a0b770ede4e98313dd57b5bbaffcd2e29286a55b1fbd3f2b9c.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/99d46231870456a0b770ede4e98313dd57b5bbaffcd2e29286a55b1fbd3f2b9c.jpg)

![b554deb7630931973fac508028aa872d8c53fd6064058a9d2be9e3a71d837e8e.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/b554deb7630931973fac508028aa872d8c53fd6064058a9d2be9e3a71d837e8e.jpg)

![b7afe90f5455ae3563d389b6b0174cf47836ea02928cc87f07b91cf3b5a45b60.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/b7afe90f5455ae3563d389b6b0174cf47836ea02928cc87f07b91cf3b5a45b60.jpg)

![eec8d969da22f75cd1c9cbacfb4f16b11f53da2e2a11d28c3fc2f4ffb273dd61.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/eec8d969da22f75cd1c9cbacfb4f16b11f53da2e2a11d28c3fc2f4ffb273dd61.jpg)

![fb592684af094ae46fe23b5ae46fb33fbbfd4f43261fd8fe26c8bf4400ccfc81.jpg](../icml_results/4282_VideoLoom_%20A%20Video%20Large%20Language%20Model%20for%20Joint%20Spatial-Temporal%20Understanding/tables/fb592684af094ae46fe23b5ae46fb33fbbfd4f43261fd8fe26c8bf4400ccfc81.jpg)

<a id="paper-4240"></a>

## Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs


### Images

![364d7183ed4443dd31b4fe195f79dfd0d1b3aede050929e1c82c94ed9dfc0d0f.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/images/364d7183ed4443dd31b4fe195f79dfd0d1b3aede050929e1c82c94ed9dfc0d0f.jpg)

![37d898ab385977e4555e7354e892a9f398847ad5ce54c6091be822fd3678bcf8.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/images/37d898ab385977e4555e7354e892a9f398847ad5ce54c6091be822fd3678bcf8.jpg)

![3bb61546591d17e4bc49583b6d7d79cdf7f1bd34e8ae46ead4b9ef1a497ba688.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/images/3bb61546591d17e4bc49583b6d7d79cdf7f1bd34e8ae46ead4b9ef1a497ba688.jpg)

![9127361e399ae0d9dc5f2a9afc24495e8da6169a90a23bb96a8e57af6c7c7cf8.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/images/9127361e399ae0d9dc5f2a9afc24495e8da6169a90a23bb96a8e57af6c7c7cf8.jpg)

![b33f4aaf8152598fd8b76b705b00467fc737dec5f0bc3e59bade2c0b9804150e.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/images/b33f4aaf8152598fd8b76b705b00467fc737dec5f0bc3e59bade2c0b9804150e.jpg)

![d06ce890d40e03f878397d1ed3f68dc43765ff25a12147e9f5d8056eeb280e09.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/images/d06ce890d40e03f878397d1ed3f68dc43765ff25a12147e9f5d8056eeb280e09.jpg)

### Tables

![0c373d38d7611723c788c4e094e7b5d94b4f627561a03a8bc0cb2251dca4df62.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/0c373d38d7611723c788c4e094e7b5d94b4f627561a03a8bc0cb2251dca4df62.jpg)

![1296246459befc723b46e4b6a1e5a922ed61e39e13c569b7b61020b76b2396b7.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/1296246459befc723b46e4b6a1e5a922ed61e39e13c569b7b61020b76b2396b7.jpg)

![2d39ce6bbd1cb9e136fb146dbda66fc6b0eaba75255140ae60057f66e47250e3.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/2d39ce6bbd1cb9e136fb146dbda66fc6b0eaba75255140ae60057f66e47250e3.jpg)

![835534d87326884f193669c3132866f4a9d2e0eb6f944047087b1c769fad1d31.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/835534d87326884f193669c3132866f4a9d2e0eb6f944047087b1c769fad1d31.jpg)

![c059cac07ef46e848fc5d14d98eb1d2ffff6d4b76b3d74decfe5327db6cd987a.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/c059cac07ef46e848fc5d14d98eb1d2ffff6d4b76b3d74decfe5327db6cd987a.jpg)

![d7025845fc8b1326f39d043b8aaea773a5045a105c3b288a593e1aadbde7f39f.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/d7025845fc8b1326f39d043b8aaea773a5045a105c3b288a593e1aadbde7f39f.jpg)

![de77a2102b3e131f65af7eb26097b1784637dcefe3b35ecf46fd60b521bb8191.jpg](../icml_results/4283_Learning%20to%20Move%20Before%20Learning%20to%20Do_%20Task-Agnostic%20pretraining%20for%20VLAs/tables/de77a2102b3e131f65af7eb26097b1784637dcefe3b35ecf46fd60b521bb8191.jpg)

<a id="paper-4241"></a>

## Think Twice Before You Act: Protecting LLM Agents Against Tool Description Poisoning via Isolated Planning


### Images

![227ee3384bad4a09f035f35e5a9ef03f804557dbe3b503576f95a86f195c7479.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/images/227ee3384bad4a09f035f35e5a9ef03f804557dbe3b503576f95a86f195c7479.jpg)

![86bd80eb7a874dbc56dc3cd70576e62e98a8f40723136e36e8b6b8818459b9eb.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/images/86bd80eb7a874dbc56dc3cd70576e62e98a8f40723136e36e8b6b8818459b9eb.jpg)

![bc19ff42b27a634a1a6ba1f192f24c4b7e90cda58c0b74cde81f7174e1b486af.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/images/bc19ff42b27a634a1a6ba1f192f24c4b7e90cda58c0b74cde81f7174e1b486af.jpg)

![ecc4122817b7d7fbee6e16a4249e63f072a6e4294819df672a01c5e8a083dfea.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/images/ecc4122817b7d7fbee6e16a4249e63f072a6e4294819df672a01c5e8a083dfea.jpg)

![f54bf063b6708e0d23f1b35b8c566bef7207847546ca86eb6b1b13edca400cb4.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/images/f54bf063b6708e0d23f1b35b8c566bef7207847546ca86eb6b1b13edca400cb4.jpg)

### Tables

![09f7d79b87e23e3edb931ffdd6ae7caf3d2a1ebfc06c81918c3888e08d1b9236.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/09f7d79b87e23e3edb931ffdd6ae7caf3d2a1ebfc06c81918c3888e08d1b9236.jpg)

![10b1a3752eddc1702224ccc0bcf820b1a4cb8576096e710f9686479bb6b418b9.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/10b1a3752eddc1702224ccc0bcf820b1a4cb8576096e710f9686479bb6b418b9.jpg)

![22cb3fc579cca1cc5e39d28dce1c445480442bf76a02d3f3560bab2242ce8d22.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/22cb3fc579cca1cc5e39d28dce1c445480442bf76a02d3f3560bab2242ce8d22.jpg)

![2c57a5a3b9a8266ad61642803dbd5f53f6bdf8d56a3586e85f76e56063ee4ae8.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/2c57a5a3b9a8266ad61642803dbd5f53f6bdf8d56a3586e85f76e56063ee4ae8.jpg)

![366aafdf512b81c1b1347e4c1675bf3dca49aaf898fe26e485b53d11fde37ce0.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/366aafdf512b81c1b1347e4c1675bf3dca49aaf898fe26e485b53d11fde37ce0.jpg)

![63fcd7f1940a6a02d3a417f5f7d398a912abdb4772fc97f3a9ee39e5c813d84d.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/63fcd7f1940a6a02d3a417f5f7d398a912abdb4772fc97f3a9ee39e5c813d84d.jpg)

![65fbdef450164156c59c42c3e0931470793625d79934f2baa207930414316222.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/65fbdef450164156c59c42c3e0931470793625d79934f2baa207930414316222.jpg)

![6b613354ba5468853fdc6e736e8b56c70eba181efa2145118c8438c24266cf7d.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/6b613354ba5468853fdc6e736e8b56c70eba181efa2145118c8438c24266cf7d.jpg)

![71ba20bacca97addccdacad201e48b8a0767900e9610cf27c3b2102ec19db3c1.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/71ba20bacca97addccdacad201e48b8a0767900e9610cf27c3b2102ec19db3c1.jpg)

![8686d391dd40d254bf07acc1242c6bcf00363b59abb21f3eb4f1104fa4aac8be.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/8686d391dd40d254bf07acc1242c6bcf00363b59abb21f3eb4f1104fa4aac8be.jpg)

![bd1bd41f91169533dcc9cbf10e1fd07b6cad7e749b8f2f2d7a488ffd9886f192.jpg](../icml_results/4284_Think%20Twice%20Before%20You%20Act_%20Protecting%20LLM%20Agents%20Against%20Tool%20Description%20Poisoning%20via%20Isolated%20Pl/tables/bd1bd41f91169533dcc9cbf10e1fd07b6cad7e749b8f2f2d7a488ffd9886f192.jpg)

<a id="paper-4242"></a>

## AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation


### Images

![2cacd57e33ea38701a33450506bff30287ab6ef4656356f962c386d6da84ead0.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/2cacd57e33ea38701a33450506bff30287ab6ef4656356f962c386d6da84ead0.jpg)

![3ca85e91c2961535c16d1ce31ac1f3eae52c32187999e99593a430cc7ca1d182.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/3ca85e91c2961535c16d1ce31ac1f3eae52c32187999e99593a430cc7ca1d182.jpg)

![686f6098f0caf25ecdd745be5e003f5cc6a46ef5bac10b59e0b3d19f15029a09.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/686f6098f0caf25ecdd745be5e003f5cc6a46ef5bac10b59e0b3d19f15029a09.jpg)

![8698bd7a5d26032e448cbda5cd4e06528f2813d4131a6fadbe7ea02b0472d0b2.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/8698bd7a5d26032e448cbda5cd4e06528f2813d4131a6fadbe7ea02b0472d0b2.jpg)

![917600699460bbc9b34a95520250922ff168908b6a1bb543f4e56165a8a2ec32.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/917600699460bbc9b34a95520250922ff168908b6a1bb543f4e56165a8a2ec32.jpg)

![b84c5634170ea427fcd29de3fd8e6e67a0d804be0233d17f085872af9a33a705.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/b84c5634170ea427fcd29de3fd8e6e67a0d804be0233d17f085872af9a33a705.jpg)

![f5481b262018206cff2fdb49a96dd62619e0d285350c21c172f3bf28d6ec2f22.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/images/f5481b262018206cff2fdb49a96dd62619e0d285350c21c172f3bf28d6ec2f22.jpg)

### Tables

![1e386ac9fc721120610c6d37d7e5776cb9fd1fac7124bd73ea59a5e8c8035df3.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/1e386ac9fc721120610c6d37d7e5776cb9fd1fac7124bd73ea59a5e8c8035df3.jpg)

![1fc7cedb3622df02dd777197f61e2f022d703e41e50397b8c32660f578899d16.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/1fc7cedb3622df02dd777197f61e2f022d703e41e50397b8c32660f578899d16.jpg)

![33e65484cd73cb477cba1a6c37cde663ceaa0605fad87a8a841b613588f6c053.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/33e65484cd73cb477cba1a6c37cde663ceaa0605fad87a8a841b613588f6c053.jpg)

![45597a41d739394931789413b1f8021af10837c1ba404ecd2224ee03dfe12373.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/45597a41d739394931789413b1f8021af10837c1ba404ecd2224ee03dfe12373.jpg)

![65a86c47d5fdf7f93c15df9d84f25d92d483b8bce078e137f5df038443992177.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/65a86c47d5fdf7f93c15df9d84f25d92d483b8bce078e137f5df038443992177.jpg)

![a9a67405e5c0bea73ae54fbf4be4f859b8682600a06cbf961101a44ab5b3e324.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/a9a67405e5c0bea73ae54fbf4be4f859b8682600a06cbf961101a44ab5b3e324.jpg)

![b3158cf4c987d48f38a8069cac55a2c8606241dcda7029ec4a244c5b335ce04f.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/b3158cf4c987d48f38a8069cac55a2c8606241dcda7029ec4a244c5b335ce04f.jpg)

![c4b35073b389b39741a8ea884fc500367cae7684db69fe8d04f01305a94ba000.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/c4b35073b389b39741a8ea884fc500367cae7684db69fe8d04f01305a94ba000.jpg)

![eb35cb5ec09464ff042e7cc5871270d4db926486da1670fe525b2249cb983d78.jpg](../icml_results/4285_AgentSelect_%20Benchmark%20for%20Narrative%20Query-to-Agent%20Recommendation/tables/eb35cb5ec09464ff042e7cc5871270d4db926486da1670fe525b2249cb983d78.jpg)

<a id="paper-4243"></a>

## Training Deep Spiking Neural Networks without Normalization


### Images

![37f323b1b3bdfd8f813beec97f0a5bddac221e8c4f1e88d97433242de18a73c1.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/37f323b1b3bdfd8f813beec97f0a5bddac221e8c4f1e88d97433242de18a73c1.jpg)

![3a2bb971cf1b38a4fbbb7bab84673bacef1cc518a6c39ae448fe3875d1694068.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/3a2bb971cf1b38a4fbbb7bab84673bacef1cc518a6c39ae448fe3875d1694068.jpg)

![572200ab247716d8407c731ab5be574764e710693bb5ca15f01af3553dfdf103.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/572200ab247716d8407c731ab5be574764e710693bb5ca15f01af3553dfdf103.jpg)

![57cd7f64242026d03cb515cb9e2092ed3c3a4558bf5b2c2e7538f8582c2559b9.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/57cd7f64242026d03cb515cb9e2092ed3c3a4558bf5b2c2e7538f8582c2559b9.jpg)

![c45aa251e6106748a662ca4266ec8624667bb39297a1fe3e40b4a5c93796ff31.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/c45aa251e6106748a662ca4266ec8624667bb39297a1fe3e40b4a5c93796ff31.jpg)

![d1129adf3a4f13dfb77122843d7f8bbbb524c511ee9b251e63040055ed576452.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/d1129adf3a4f13dfb77122843d7f8bbbb524c511ee9b251e63040055ed576452.jpg)

![e36488cbb695a8bf03286f72da6bf0364877de1191fd327246a2462752c13529.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/images/e36488cbb695a8bf03286f72da6bf0364877de1191fd327246a2462752c13529.jpg)

### Tables

![0f83dfe0bc6f9040d477e1d0a028155b2b9788e8ed5169fb9e590c31414b5c22.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/0f83dfe0bc6f9040d477e1d0a028155b2b9788e8ed5169fb9e590c31414b5c22.jpg)

![747d8bd2cc4ab22b0906e983d03d536d7b1613f001a8a0a6c2d1588f1d37e56e.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/747d8bd2cc4ab22b0906e983d03d536d7b1613f001a8a0a6c2d1588f1d37e56e.jpg)

![b278317af727b974b48edb2e8c7a0772e5a4f919873cb27f0588552ea79729f0.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/b278317af727b974b48edb2e8c7a0772e5a4f919873cb27f0588552ea79729f0.jpg)

![ddcc6f74413c80b68461170cf9c27e314bda3b37721ef7eb3fb0e7825cc85e8b.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/ddcc6f74413c80b68461170cf9c27e314bda3b37721ef7eb3fb0e7825cc85e8b.jpg)

![ea1e335aa14bca5d7b6b686db5617f7b25ae8e3b14b0ff395a8856d81e987779.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/ea1e335aa14bca5d7b6b686db5617f7b25ae8e3b14b0ff395a8856d81e987779.jpg)

![ece20fdd53bd49b05218c30d8443bbc45ce23a58e5c5eaaa8962889705ea26eb.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/ece20fdd53bd49b05218c30d8443bbc45ce23a58e5c5eaaa8962889705ea26eb.jpg)

![ee936245f6933e217f8b4746e734fbeb130da6fcee5767690ae7d2d5760ce22a.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/ee936245f6933e217f8b4746e734fbeb130da6fcee5767690ae7d2d5760ce22a.jpg)

![f9e9dd3330cc65a26a4150d7ec23a078e7a0fd873c3ef9f081d9d42537ff1f44.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/f9e9dd3330cc65a26a4150d7ec23a078e7a0fd873c3ef9f081d9d42537ff1f44.jpg)

![fb888a21097149b179acf74abd0702468f90fc9ee32be31efaabad1502893f71.jpg](../icml_results/4286_Training%20Deep%20Spiking%20Neural%20Networks%20without%20Normalization/tables/fb888a21097149b179acf74abd0702468f90fc9ee32be31efaabad1502893f71.jpg)

<a id="paper-4244"></a>

## AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction


### Images

![37fd03394033f7b492143f5113404cde22c044cae15901da0270ee30e4afc8e3.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/images/37fd03394033f7b492143f5113404cde22c044cae15901da0270ee30e4afc8e3.jpg)

![5a417d575699e07e1cbd19202e6298365009e88f850bb05c495dee36bed5f199.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/images/5a417d575699e07e1cbd19202e6298365009e88f850bb05c495dee36bed5f199.jpg)

![6703575015c2947e78e658480c32d3af386b8706f902ca7e1d18b02b03111074.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/images/6703575015c2947e78e658480c32d3af386b8706f902ca7e1d18b02b03111074.jpg)

![9d02af5f789601ae7c38356124c120991921a3b52a03042b737e1663ee69f381.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/images/9d02af5f789601ae7c38356124c120991921a3b52a03042b737e1663ee69f381.jpg)

![d15aa19bbdd492c030f05feaa03a9c8d8eab0adbc14fdff1d899f2b6000b7ac7.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/images/d15aa19bbdd492c030f05feaa03a9c8d8eab0adbc14fdff1d899f2b6000b7ac7.jpg)

### Tables

![04391f5b43ff00a2ff8bb30ce92ae215b469b2940460621395d0f886d189fc78.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/tables/04391f5b43ff00a2ff8bb30ce92ae215b469b2940460621395d0f886d189fc78.jpg)

![9281e636682252fab59b786620a3b6c540dd6cb212ef0630fc01fdfdb8bc9998.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/tables/9281e636682252fab59b786620a3b6c540dd6cb212ef0630fc01fdfdb8bc9998.jpg)

![c1e66cad095f011e26b852e22f0438e8a269c1c0ee1f0d4e4ba6e06c186951bd.jpg](../icml_results/4287_AgentXRay_%20White-Boxing%20Agentic%20Systems%20via%20Workflow%20Reconstruction/tables/c1e66cad095f011e26b852e22f0438e8a269c1c0ee1f0d4e4ba6e06c186951bd.jpg)

<a id="paper-4245"></a>

## Structured Multi-modal Graph Disentanglement for Psychiatric Diagnosis


### Images

![3268f96ce85c985f9e9050aeb6055ccfdaff8f690734b21bbdf119465e948cc0.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/3268f96ce85c985f9e9050aeb6055ccfdaff8f690734b21bbdf119465e948cc0.jpg)

![423e687addb0dbcca312c27204a4ab0a4f0c79fcbe877b0e20b49ea2fa4bbb3a.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/423e687addb0dbcca312c27204a4ab0a4f0c79fcbe877b0e20b49ea2fa4bbb3a.jpg)

![621a49dbd480a23c61cf202286148d83513af41f6985173586a12a776dedcbdc.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/621a49dbd480a23c61cf202286148d83513af41f6985173586a12a776dedcbdc.jpg)

![62d5afd14daaf74e3d260fb742289254b1e934e99b8f12308c2614629f2880b5.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/62d5afd14daaf74e3d260fb742289254b1e934e99b8f12308c2614629f2880b5.jpg)

![6c0dc3df53fd538e51abfc1e3702715a7c0de2f05e71ebb05d068d5996ed153a.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/6c0dc3df53fd538e51abfc1e3702715a7c0de2f05e71ebb05d068d5996ed153a.jpg)

![b17c8a8160915b31c3821e7e66414424640fb7eb999a85a1ffad3821defa0619.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/b17c8a8160915b31c3821e7e66414424640fb7eb999a85a1ffad3821defa0619.jpg)

![b565a0190a646894f69b6543081e369ce0ef6484aefc4564e35c9fefbf17815a.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/b565a0190a646894f69b6543081e369ce0ef6484aefc4564e35c9fefbf17815a.jpg)

![f54281fb5e662d3a62ddfc2ae191cc17f7c6778f3f461fc8b4f3146613324862.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/images/f54281fb5e662d3a62ddfc2ae191cc17f7c6778f3f461fc8b4f3146613324862.jpg)

### Tables

![12bfc134be5cef52d574ae480bd11c6ae5bce61d6cd991294f3524eeb2defe7c.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/12bfc134be5cef52d574ae480bd11c6ae5bce61d6cd991294f3524eeb2defe7c.jpg)

![1e12576fb59f1aeb98a2fe0ccffd630b43d253368d54b80e48379422ff125612.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/1e12576fb59f1aeb98a2fe0ccffd630b43d253368d54b80e48379422ff125612.jpg)

![6606c706e7fbe6be045dcfab274aef9d79bf2f412eecbefe83bf7c3e4e260b08.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/6606c706e7fbe6be045dcfab274aef9d79bf2f412eecbefe83bf7c3e4e260b08.jpg)

![b5ce3b4095bfab7e862b15077d1e6d3f8f4e1e5593ed170a8abb5a6564ff2089.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/b5ce3b4095bfab7e862b15077d1e6d3f8f4e1e5593ed170a8abb5a6564ff2089.jpg)

![cafda235f25dae66100d9cf6621866985668e79f4d050faf1cee501cea10ba0f.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/cafda235f25dae66100d9cf6621866985668e79f4d050faf1cee501cea10ba0f.jpg)

![ddea794e027ec6512abda658ba3ff15cd8cf10c60cbe5b5eed7d8b8ff0e186d6.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/ddea794e027ec6512abda658ba3ff15cd8cf10c60cbe5b5eed7d8b8ff0e186d6.jpg)

![e85477f3f1b39b624acec880d4a8f3da333bf6efebd3ec50c11fb10145688d44.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/e85477f3f1b39b624acec880d4a8f3da333bf6efebd3ec50c11fb10145688d44.jpg)

![f4128258c25b19b0f08cd0ccbf768828bd8cd4e5627e0ea6fea16a5fd5129e8e.jpg](../icml_results/4288_Structured%20Multi-modal%20Graph%20Disentanglement%20for%20Psychiatric%20Diagnosis/tables/f4128258c25b19b0f08cd0ccbf768828bd8cd4e5627e0ea6fea16a5fd5129e8e.jpg)

<a id="paper-4246"></a>

## $τ$-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge


### Images

![80c35d1cf401336477ad26f49de5e3f3100234021e6cba4699e6830c6ef28b53.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/images/80c35d1cf401336477ad26f49de5e3f3100234021e6cba4699e6830c6ef28b53.jpg)

![dbe71a23296f3d448187f447591411756376254e5de1670533fffab42b0353ce.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/images/dbe71a23296f3d448187f447591411756376254e5de1670533fffab42b0353ce.jpg)

![e211843628a111bfc0b0234b134e4969632fb18fc1f15abc84d5d633aabdc346.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/images/e211843628a111bfc0b0234b134e4969632fb18fc1f15abc84d5d633aabdc346.jpg)

![f39380482d10b05667db292cf8a84241e8f893eb21c467089c8700164f37c7d3.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/images/f39380482d10b05667db292cf8a84241e8f893eb21c467089c8700164f37c7d3.jpg)

### Tables

![04c19f181c927309bd69c8209f68051575c62514ee6cc1cc68cb4475e4b27872.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/04c19f181c927309bd69c8209f68051575c62514ee6cc1cc68cb4475e4b27872.jpg)

![157eb05c364444c2b1d2adfd7b3d526488efcb55c244291a10a58fbaf5805f99.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/157eb05c364444c2b1d2adfd7b3d526488efcb55c244291a10a58fbaf5805f99.jpg)

![1d469167336e2beebcdef02ee374e63341cfeba0abce95be52fa12d2e8a510d2.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/1d469167336e2beebcdef02ee374e63341cfeba0abce95be52fa12d2e8a510d2.jpg)

![2a1a6528a9fb64f740c56791e6eaa9e0de0f9df1cb684fc7390f99e5e3dcdee9.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/2a1a6528a9fb64f740c56791e6eaa9e0de0f9df1cb684fc7390f99e5e3dcdee9.jpg)

![2fe43aaf2e498487bdee1fe7a9c4f856f5a442ef3371e77c0562a03dc35439c5.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/2fe43aaf2e498487bdee1fe7a9c4f856f5a442ef3371e77c0562a03dc35439c5.jpg)

![300ada805a349a6ae6e83e5c92033b84fdb586a22b9952bd7000bb1e3e04a274.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/300ada805a349a6ae6e83e5c92033b84fdb586a22b9952bd7000bb1e3e04a274.jpg)

![4e76430a78c867c7fa26fb76e30d2f6458c6b25412f75452c0793a765f1b44d9.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/4e76430a78c867c7fa26fb76e30d2f6458c6b25412f75452c0793a765f1b44d9.jpg)

![57b9d8af2212481f994b3092196e65e3dd028ac1c10ecdfc9186e554e44934ad.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/57b9d8af2212481f994b3092196e65e3dd028ac1c10ecdfc9186e554e44934ad.jpg)

![7ded68d224e2ef64b5b71b1c718a2dca9823c47602eb8e2e90aef1348cc591ac.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/7ded68d224e2ef64b5b71b1c718a2dca9823c47602eb8e2e90aef1348cc591ac.jpg)

![8d2366e143b866d78bf234d9253a4a79c0bcf01eeae305b891dfac0bfb60bf06.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/8d2366e143b866d78bf234d9253a4a79c0bcf01eeae305b891dfac0bfb60bf06.jpg)

![90486bc72742e281c4e10397d9e94d42d4f02536e125fa391fdc5fe451849b53.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/90486bc72742e281c4e10397d9e94d42d4f02536e125fa391fdc5fe451849b53.jpg)

![9e58ff3686bf7b8cf247f40894c39b4ea89f3883149f7ec1f5ec8c160c00b864.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/9e58ff3686bf7b8cf247f40894c39b4ea89f3883149f7ec1f5ec8c160c00b864.jpg)

![bbbd4a2b2ac5425f97707b0c528aa61ebdbcb1f2ef818c41a31c5e4b2597be2b.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/bbbd4a2b2ac5425f97707b0c528aa61ebdbcb1f2ef818c41a31c5e4b2597be2b.jpg)

![dfbb7bc5afbcf6de6bd0f29a3806b3a7db1ec1cb8d9decc5f519acf5cea552f7.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/dfbb7bc5afbcf6de6bd0f29a3806b3a7db1ec1cb8d9decc5f519acf5cea552f7.jpg)

![ffcccc8c75d4eac490760f9c99e4a479289cd481df10156ba81c7e22d128f311.jpg](../icml_results/4289_%24%CF%84%24-Knowledge_%20Evaluating%20Conversational%20Agents%20over%20Unstructured%20Knowledge/tables/ffcccc8c75d4eac490760f9c99e4a479289cd481df10156ba81c7e22d128f311.jpg)

<a id="paper-4247"></a>

## CGRiC: Compositional Risk Certification for Structured LLM Outputs


### Images

![6d84a917946ca5c89db867a7aa06de8fd786654ef5abb90bf3261989955146b5.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/images/6d84a917946ca5c89db867a7aa06de8fd786654ef5abb90bf3261989955146b5.jpg)

![c649c2ee2e88123627c61955d0a31696608dc71f0d6049c6dac22e7fe574320c.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/images/c649c2ee2e88123627c61955d0a31696608dc71f0d6049c6dac22e7fe574320c.jpg)

### Tables

![008f3019ac0f3dc8f57d2e035e5ec5557dfe25a5c67980ca5b67104cb373fbeb.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/008f3019ac0f3dc8f57d2e035e5ec5557dfe25a5c67980ca5b67104cb373fbeb.jpg)

![03db339d3fc42c0a746916ba8e5e0ae88d41fc0b268a46c95220f3c3e3226592.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/03db339d3fc42c0a746916ba8e5e0ae88d41fc0b268a46c95220f3c3e3226592.jpg)

![1a6f6f379f6e88a58f0987dd8cd1c03271866e2743dcf92d5d07aa60b588928d.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/1a6f6f379f6e88a58f0987dd8cd1c03271866e2743dcf92d5d07aa60b588928d.jpg)

![1ba83bff7bce44e83d1557e31b9f19e150d6d0a0d9a57332c18e232a55f5ec15.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/1ba83bff7bce44e83d1557e31b9f19e150d6d0a0d9a57332c18e232a55f5ec15.jpg)

![257b07f2bfd8ca01972be9627bd47632acbc40b1d097b5cc915d7309e1fc01ed.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/257b07f2bfd8ca01972be9627bd47632acbc40b1d097b5cc915d7309e1fc01ed.jpg)

![26045a6d575f2314a0d057e6f885d8092efa6a36e6a4dca217d750ba825fad28.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/26045a6d575f2314a0d057e6f885d8092efa6a36e6a4dca217d750ba825fad28.jpg)

![2995745d9c5d0f295f8138e8bbd00e19a001e64b775e3f1dcc0b90d9080682c8.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/2995745d9c5d0f295f8138e8bbd00e19a001e64b775e3f1dcc0b90d9080682c8.jpg)

![36fddcd6c6d6d38bd07132bb8f70fb883c144468398a81ce8ff266720ebacb44.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/36fddcd6c6d6d38bd07132bb8f70fb883c144468398a81ce8ff266720ebacb44.jpg)

![394e1d71f3a71d6878ded32972884068e95eb94b7393669183c278215662683b.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/394e1d71f3a71d6878ded32972884068e95eb94b7393669183c278215662683b.jpg)

![4508f937337b440693d888b0db0a648f4e9ae0753a0cce97e16690241c21a63e.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/4508f937337b440693d888b0db0a648f4e9ae0753a0cce97e16690241c21a63e.jpg)

![577e270fb0cf5704e0248746081f8162371db886457eb1c74e7e61f6ae7b55fb.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/577e270fb0cf5704e0248746081f8162371db886457eb1c74e7e61f6ae7b55fb.jpg)

![5994dcd18a3dc6654fc2ecc22c54c705a7c2628007621a81225e8fc082a2d246.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/5994dcd18a3dc6654fc2ecc22c54c705a7c2628007621a81225e8fc082a2d246.jpg)

![65005e964deb491ea50a8ea030d06eb01952cc7288f47f5d076fb82bd6154896.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/65005e964deb491ea50a8ea030d06eb01952cc7288f47f5d076fb82bd6154896.jpg)

![6d75be1d94d32bdcb460222fe1fa847ef0b95b548ef4fb25badebc5b578e0072.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/6d75be1d94d32bdcb460222fe1fa847ef0b95b548ef4fb25badebc5b578e0072.jpg)

![6e3c0ea7e7cda37b7d58ed60efee0551f53b8bf9e7595932f45855937db230c7.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/6e3c0ea7e7cda37b7d58ed60efee0551f53b8bf9e7595932f45855937db230c7.jpg)

![6fd512b4f07cfa0db25f36c579807b499fdf7aabdf1a55adcc80599e3be6f595.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/6fd512b4f07cfa0db25f36c579807b499fdf7aabdf1a55adcc80599e3be6f595.jpg)

![75ba1dcbebbf229936737304a98ad0515fa6d22a75018844d7687a1abe42d92d.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/75ba1dcbebbf229936737304a98ad0515fa6d22a75018844d7687a1abe42d92d.jpg)

![786463ac64c5ae8ab057dc568f9c415ded5f5ca62473790b4cefe0238a51d522.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/786463ac64c5ae8ab057dc568f9c415ded5f5ca62473790b4cefe0238a51d522.jpg)

![788e70dd7b4ad00a8a5bf5df1820ef49ab3a7ef31cfbea28e66e374b4f2ac98a.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/788e70dd7b4ad00a8a5bf5df1820ef49ab3a7ef31cfbea28e66e374b4f2ac98a.jpg)

![78eea6259f15b5d10fabb3dd7f562d7724c559107d8885ce1a0707a821947aac.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/78eea6259f15b5d10fabb3dd7f562d7724c559107d8885ce1a0707a821947aac.jpg)

![7bff9093ab1232d8da1bfc21c13467862680916f121d549f455850d9a07b5fc4.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/7bff9093ab1232d8da1bfc21c13467862680916f121d549f455850d9a07b5fc4.jpg)

![7cd3ba432775e25ec763b74320b472e5f7239508f158457e6b35d3fa16c7f7a3.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/7cd3ba432775e25ec763b74320b472e5f7239508f158457e6b35d3fa16c7f7a3.jpg)

![7e2bd6c07a7a87772c2d392899123bc3d59eaa266a2b0bd80b17b85489e257d4.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/7e2bd6c07a7a87772c2d392899123bc3d59eaa266a2b0bd80b17b85489e257d4.jpg)

![8437352625427c095c91478ba291574bab9ea1cfbdbbc75b84532128b62db6b4.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/8437352625427c095c91478ba291574bab9ea1cfbdbbc75b84532128b62db6b4.jpg)

![88d04371094adac6501f0c958d17a73f012214d254f9d811911f9e5d6a69d6df.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/88d04371094adac6501f0c958d17a73f012214d254f9d811911f9e5d6a69d6df.jpg)

![9a93fc3c31b72eef8124a8fcaebb74ee3af8a4e75380dd86f3b6736435919fc1.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/9a93fc3c31b72eef8124a8fcaebb74ee3af8a4e75380dd86f3b6736435919fc1.jpg)

![a5a2905a669f083e83d07229d3c4ae8735f83b9203036ab64d380045667f4567.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/a5a2905a669f083e83d07229d3c4ae8735f83b9203036ab64d380045667f4567.jpg)

![a5cd5ed57287259ab56c2fbb2115ae3fd41253e3eb71acfb40102ec5f7ffa4ae.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/a5cd5ed57287259ab56c2fbb2115ae3fd41253e3eb71acfb40102ec5f7ffa4ae.jpg)

![b3dafdf9882d100047b2b5d05742d60ea9623c92e7bef3b13a605485cdc6d529.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/b3dafdf9882d100047b2b5d05742d60ea9623c92e7bef3b13a605485cdc6d529.jpg)

![c470ab58ab6d4988a9086fc8e18e58d2cd1a9ab17583891fd299a298230b5a16.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/c470ab58ab6d4988a9086fc8e18e58d2cd1a9ab17583891fd299a298230b5a16.jpg)

![c556c67c79890f92e188b10473c17eeea817d6f254cb2746b1defe88fe9f086d.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/c556c67c79890f92e188b10473c17eeea817d6f254cb2746b1defe88fe9f086d.jpg)

![cc464c3c8d13c3783b54b458ae0f4734e7d6ce3548b619a175b859aa613d59d8.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/cc464c3c8d13c3783b54b458ae0f4734e7d6ce3548b619a175b859aa613d59d8.jpg)

![cdd97e93b57eb8f79a4aaea9b261d74f6b186d15bdb39b814a2adde6f5ce10f2.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/cdd97e93b57eb8f79a4aaea9b261d74f6b186d15bdb39b814a2adde6f5ce10f2.jpg)

![d1e5c45aa55031962c41b0818b1e3af5867b579af55a381814c240c32d5bd0cf.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/d1e5c45aa55031962c41b0818b1e3af5867b579af55a381814c240c32d5bd0cf.jpg)

![d42ea4ff2120c12b0c33c593da7b0b37aca77a3f7c8fa3da5a35b86088902291.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/d42ea4ff2120c12b0c33c593da7b0b37aca77a3f7c8fa3da5a35b86088902291.jpg)

![de54af6860b5ce52930837cecaa74a9e03e08f6bb26aa9f0617085a40648c148.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/de54af6860b5ce52930837cecaa74a9e03e08f6bb26aa9f0617085a40648c148.jpg)

![deacf7efcb2d2f355787dc6d3a15bf4cbb881922e294bb24bc208b35c50b7d6f.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/deacf7efcb2d2f355787dc6d3a15bf4cbb881922e294bb24bc208b35c50b7d6f.jpg)

![e22421f79819a55d65a32b364501c31145bbc0db29bc2117e8f04b476d027f0f.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/e22421f79819a55d65a32b364501c31145bbc0db29bc2117e8f04b476d027f0f.jpg)

![f606673b9caaeb89f3800e190e91a1535986ace2adb77ff6824e7737be02a58d.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/f606673b9caaeb89f3800e190e91a1535986ace2adb77ff6824e7737be02a58d.jpg)

![fb20639906b57a97df3ae3201c25e0cc283c451f588df2f431d130da64c542aa.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/fb20639906b57a97df3ae3201c25e0cc283c451f588df2f431d130da64c542aa.jpg)

![fd6d83d59657f745722104526aceddd9dae74c180389ba7f3215a27eb52ac00b.jpg](../icml_results/4290_CGRiC_%20Compositional%20Risk%20Certification%20for%20Structured%20LLM%20Outputs/tables/fd6d83d59657f745722104526aceddd9dae74c180389ba7f3215a27eb52ac00b.jpg)

<a id="paper-4248"></a>

## CalPro: Prior-Aware Evidential Conformal Prediction with Structure-Aware Sensitivity Bounds for Protein Structures


### Images

![025e0bbc566c6c404624cad172f00b576301cdeacfb8b61967cde60595a64be9.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/025e0bbc566c6c404624cad172f00b576301cdeacfb8b61967cde60595a64be9.jpg)

![239ec13b68fb6660b8da1cd4502e7208b870baa6c995eaf2f96ae8d2aace83e1.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/239ec13b68fb6660b8da1cd4502e7208b870baa6c995eaf2f96ae8d2aace83e1.jpg)

![4a29c11aa4b28dd386b2999eaf0377c9c38a6d2fc1d14f6bf50356e70c655023.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/4a29c11aa4b28dd386b2999eaf0377c9c38a6d2fc1d14f6bf50356e70c655023.jpg)

![559973ecfe36e1da4f7dc2ee1886bde58c08950a8063741fde2b40af7244a354.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/559973ecfe36e1da4f7dc2ee1886bde58c08950a8063741fde2b40af7244a354.jpg)

![65f2eecd431e769d722ff965bc0306923ce160d4c3b581151fc612c05ec2222e.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/65f2eecd431e769d722ff965bc0306923ce160d4c3b581151fc612c05ec2222e.jpg)

![bf8e2a08aaee17db321c4aa592e86fe1eae513a0ccffd9dfabb99828b10fce19.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/bf8e2a08aaee17db321c4aa592e86fe1eae513a0ccffd9dfabb99828b10fce19.jpg)

![c6881174d17c6d29608150211e57f70626d25d3f3ab73cb968c79ace96acb7b8.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/images/c6881174d17c6d29608150211e57f70626d25d3f3ab73cb968c79ace96acb7b8.jpg)

### Tables

![0665be7d7aa247a4bb629bf1d48a2f3970b075e94296ef04fe94130850dcc542.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/0665be7d7aa247a4bb629bf1d48a2f3970b075e94296ef04fe94130850dcc542.jpg)

![0cba079e95822e77cfb51283e0ff6d60b0506af8256d60796f0590a92607a61a.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/0cba079e95822e77cfb51283e0ff6d60b0506af8256d60796f0590a92607a61a.jpg)

![1bfbe66c17b362eafc10955663d2984022d66c0ab27342a1814fd3e4ebc268a1.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/1bfbe66c17b362eafc10955663d2984022d66c0ab27342a1814fd3e4ebc268a1.jpg)

![2b8b423cf990d456a5666faf8033c311380548d320a34b9a5440e244ce463237.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/2b8b423cf990d456a5666faf8033c311380548d320a34b9a5440e244ce463237.jpg)

![3304822063d3b29e9c7e05962a86913e5a6a8bdba26231977080a5e295c95067.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/3304822063d3b29e9c7e05962a86913e5a6a8bdba26231977080a5e295c95067.jpg)

![4234658a696c0b8d92630206bda31ea013488e51d023deb7df1ce28e4e845747.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/4234658a696c0b8d92630206bda31ea013488e51d023deb7df1ce28e4e845747.jpg)

![45bc1e617bf0e70e73d9446e7dbdf0f4c5d040533a60bb83d8864af94ee848e2.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/45bc1e617bf0e70e73d9446e7dbdf0f4c5d040533a60bb83d8864af94ee848e2.jpg)

![5259d453508632a9e52ef57f0f747c170acf3c527baf038e4317e0e82472f863.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/5259d453508632a9e52ef57f0f747c170acf3c527baf038e4317e0e82472f863.jpg)

![56dd3da8fd7f8556114b207785090bd02be1321259a14fb084eed6ec72be1bc6.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/56dd3da8fd7f8556114b207785090bd02be1321259a14fb084eed6ec72be1bc6.jpg)

![6727443934a430a3d0f6c732d8f9d51fd23a04bab8d6cea013883d0b0555e331.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/6727443934a430a3d0f6c732d8f9d51fd23a04bab8d6cea013883d0b0555e331.jpg)

![8035cb9d93daf3c1ea757f196d3d421095004f885cc7289e075dc8e467548564.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/8035cb9d93daf3c1ea757f196d3d421095004f885cc7289e075dc8e467548564.jpg)

![88cffbf0e9b6adad83632fc7ca52709a70d530da0723766e02e4988a55c38752.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/88cffbf0e9b6adad83632fc7ca52709a70d530da0723766e02e4988a55c38752.jpg)

![89ae57caa9ba587cf759d94f39481243869fc50623c656f16a6ad7ebeafefc34.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/89ae57caa9ba587cf759d94f39481243869fc50623c656f16a6ad7ebeafefc34.jpg)

![8aacd53c2fc8fdde3ffc65743a445a7fd8568c16e7e8b658ec21c2a950d5a7d0.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/8aacd53c2fc8fdde3ffc65743a445a7fd8568c16e7e8b658ec21c2a950d5a7d0.jpg)

![9bcc3e4e240b0276d2e56666e732b32e87e38382e07d83b95edd45cda5cff8a0.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/9bcc3e4e240b0276d2e56666e732b32e87e38382e07d83b95edd45cda5cff8a0.jpg)

![a6d706613f44a82c82d15f0bfb2aaf93bc43795215e39e64ce046e21da675d70.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/a6d706613f44a82c82d15f0bfb2aaf93bc43795215e39e64ce046e21da675d70.jpg)

![ae7729953fb21762e896e28374b8daf88fffa1e86bdc48bba4b8aaf6017cce45.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/ae7729953fb21762e896e28374b8daf88fffa1e86bdc48bba4b8aaf6017cce45.jpg)

![b7db3cc7c37a6b7d0405d3601fa27b5f242bb3bf06e24cfcfc965a0a71181216.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/b7db3cc7c37a6b7d0405d3601fa27b5f242bb3bf06e24cfcfc965a0a71181216.jpg)

![d0b0fbb54f25f3cd03126769fa7f0b8cf25d42e29ebcc2f006b4a0cdd7544814.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/d0b0fbb54f25f3cd03126769fa7f0b8cf25d42e29ebcc2f006b4a0cdd7544814.jpg)

![d0e9b53733913e0bc44c05ba9983081fbbf339db380745eb0fbc5aab52c8ecc0.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/d0e9b53733913e0bc44c05ba9983081fbbf339db380745eb0fbc5aab52c8ecc0.jpg)

![e90f9d0077779ca6f10598606a013493a1afe4c7447757eaf25644a06f6c9c5c.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/e90f9d0077779ca6f10598606a013493a1afe4c7447757eaf25644a06f6c9c5c.jpg)

![ef4c74cf53f9a2d04a943297f394aaa407de2ad5d181d251340dcb96b0c8aae4.jpg](../icml_results/4291_CalPro_%20Prior-Aware%20Evidential%20Conformal%20Prediction%20with%20Structure-Aware%20Sensitivity%20Bounds%20for%20Prot/tables/ef4c74cf53f9a2d04a943297f394aaa407de2ad5d181d251340dcb96b0c8aae4.jpg)

<a id="paper-4249"></a>

## Certificate-Guided Pruning for Stochastic Lipschitz Optimization


### Images

![f6ca714f5a5006de1ec683eef48e696a7af97d555d9c6e646d13d6da0194ea99.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/images/f6ca714f5a5006de1ec683eef48e696a7af97d555d9c6e646d13d6da0194ea99.jpg)

### Tables

![05759d698f521e8224dacef75d0a464179bdfc4ceec33546bdc8e00805e6f4f8.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/05759d698f521e8224dacef75d0a464179bdfc4ceec33546bdc8e00805e6f4f8.jpg)

![170fb5aa10a0cb465b9fa88edb9777a25c871d1b6aca9a0dada8d292a140f340.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/170fb5aa10a0cb465b9fa88edb9777a25c871d1b6aca9a0dada8d292a140f340.jpg)

![6379443d423bd309571d180441bbb13f28afe3b2691e624dea67bf09c3f3db1f.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/6379443d423bd309571d180441bbb13f28afe3b2691e624dea67bf09c3f3db1f.jpg)

![706f407dc718efbef67b2d01b88c8378c9bf653830ea728ea1f69cc3d217c414.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/706f407dc718efbef67b2d01b88c8378c9bf653830ea728ea1f69cc3d217c414.jpg)

![720da250f653c067a59a90db32b1b47e8420ca243b63ca0d7de2a8a5d21393f2.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/720da250f653c067a59a90db32b1b47e8420ca243b63ca0d7de2a8a5d21393f2.jpg)

![737b5721ffd4ba6de8f9b97fdb6c8e3225b5d985c1638177078b0a1413ed16b7.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/737b5721ffd4ba6de8f9b97fdb6c8e3225b5d985c1638177078b0a1413ed16b7.jpg)

![7f618cc002b0a48f82d984980abf9a7751eb98b906c5678a6c4562fcd3008abe.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/7f618cc002b0a48f82d984980abf9a7751eb98b906c5678a6c4562fcd3008abe.jpg)

![985cc1d2cf9746d473fb5400f4f8b986aa3dab18dfb5c581615ea7c3d66aa6be.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/985cc1d2cf9746d473fb5400f4f8b986aa3dab18dfb5c581615ea7c3d66aa6be.jpg)

![b5bad02ae9d713da64959aa197a58ce998fcae8bb6917b00759d3c4dc7334e18.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/b5bad02ae9d713da64959aa197a58ce998fcae8bb6917b00759d3c4dc7334e18.jpg)

![b6abf88ac0d6ed74900f08bcb90581642c96566536626a1f746baddcb1a41ac7.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/b6abf88ac0d6ed74900f08bcb90581642c96566536626a1f746baddcb1a41ac7.jpg)

![de99a0292ca7836e90ec92bdfeac5495d3330624695763a8f3c472bf5591d9c6.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/de99a0292ca7836e90ec92bdfeac5495d3330624695763a8f3c472bf5591d9c6.jpg)

![df1f2e38f7f38f53bd4bf15ff23fbe58e793965296fadd225136ceb2d9387295.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/df1f2e38f7f38f53bd4bf15ff23fbe58e793965296fadd225136ceb2d9387295.jpg)

![f09b2de62e5cabc19d897f85d4221caf7f793a3eeab5dcd7fea5f6d8b0181ec2.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/f09b2de62e5cabc19d897f85d4221caf7f793a3eeab5dcd7fea5f6d8b0181ec2.jpg)

![f45ad07dbbc31d94d207957d04c5ca973cccb7547bacf50e33727784b3a7a3e4.jpg](../icml_results/4292_Certificate-Guided%20Pruning%20for%20Stochastic%20Lipschitz%20Optimization/tables/f45ad07dbbc31d94d207957d04c5ca973cccb7547bacf50e33727784b3a7a3e4.jpg)

<a id="paper-4250"></a>

## What Reward Structure Enables Efficient Sparse-Reward RL? A Proof-of-Concept with Policy-Aware Matrix Completion


### Images

![289a5f2726591a8bec19422b3025274598ba45af95e21e9ba06514f057477c0d.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/289a5f2726591a8bec19422b3025274598ba45af95e21e9ba06514f057477c0d.jpg)

![37cb9c413f211442162bdae713d107711ad67bb05b8980574dce72bbcf45f07b.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/37cb9c413f211442162bdae713d107711ad67bb05b8980574dce72bbcf45f07b.jpg)

![3852bcb07e663026094119ef6081df5d62076fb0e0a63d8b6e6e6015ed337b72.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/3852bcb07e663026094119ef6081df5d62076fb0e0a63d8b6e6e6015ed337b72.jpg)

![513a20df50636b3027b47593d9dd5cb4883d8cb8aec21ae40009c44a89f60415.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/513a20df50636b3027b47593d9dd5cb4883d8cb8aec21ae40009c44a89f60415.jpg)

![566a3d193f8f619e12fac8bb60bc80b64dd2a4ae305f87c88f4d830ec44c1f2c.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/566a3d193f8f619e12fac8bb60bc80b64dd2a4ae305f87c88f4d830ec44c1f2c.jpg)

![56f2af3ab4a058a903d599971e9f541bcae5d187d072c1990df84597220e96dc.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/56f2af3ab4a058a903d599971e9f541bcae5d187d072c1990df84597220e96dc.jpg)

![62d450dccebb5f2379da5d7878e514f0a6408bc1556038543123fba5e1a12f2c.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/62d450dccebb5f2379da5d7878e514f0a6408bc1556038543123fba5e1a12f2c.jpg)

![660008f8f24ef1645951beb15bd68fbf45691eb38bcded4f8dd82f0035b36d09.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/660008f8f24ef1645951beb15bd68fbf45691eb38bcded4f8dd82f0035b36d09.jpg)

![6a58bb8952bac8277c8a45a89b1130d1823fd86c689a8371d23c0913235f9c76.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/6a58bb8952bac8277c8a45a89b1130d1823fd86c689a8371d23c0913235f9c76.jpg)

![7994d91db2e39e1aba050af3ab985f1315a3399fa8a9dbaa25fc33ecad0685f8.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/7994d91db2e39e1aba050af3ab985f1315a3399fa8a9dbaa25fc33ecad0685f8.jpg)

![7ded19376523f34ca780b437dbdd9433b6b2055af6059aa0b14b127a3de08d70.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/7ded19376523f34ca780b437dbdd9433b6b2055af6059aa0b14b127a3de08d70.jpg)

![8f3f90a4c401ebc33645ee81046b9bac549033be878a27482cb3c3755e7416f1.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/8f3f90a4c401ebc33645ee81046b9bac549033be878a27482cb3c3755e7416f1.jpg)

![9f2918a0b8318b8d9cd7c587caf6bd829648d26b2040bf3798fb0b872d874eec.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/9f2918a0b8318b8d9cd7c587caf6bd829648d26b2040bf3798fb0b872d874eec.jpg)

![a20892b2bdef0d45b812dc767d86528ef6ef2e7a808e5af85b5cc74084f38c2c.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/a20892b2bdef0d45b812dc767d86528ef6ef2e7a808e5af85b5cc74084f38c2c.jpg)

![af1cbe2eebfd0b71ac2a4bed26e3674da78db4ad95a26cd60c9401b1e891e696.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/af1cbe2eebfd0b71ac2a4bed26e3674da78db4ad95a26cd60c9401b1e891e696.jpg)

![b8059fdc52c2aa7b502808e1f9dbe602f29a970eb2627e8b375bb4e00cf5bca3.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/b8059fdc52c2aa7b502808e1f9dbe602f29a970eb2627e8b375bb4e00cf5bca3.jpg)

![c36e91762fda0b47400334e4d8fb4b72cbb8843ba35cebfb11d5815f07edad05.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/c36e91762fda0b47400334e4d8fb4b72cbb8843ba35cebfb11d5815f07edad05.jpg)

![e1a2a7e41b4daaa527e91c9581914b3da748344daed9f57d4409e796ef00cfc8.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/e1a2a7e41b4daaa527e91c9581914b3da748344daed9f57d4409e796ef00cfc8.jpg)

![e22965e30ec856b80bea49e23e55adb9119a385d8b3688db3c9024b1bf5afa3f.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/e22965e30ec856b80bea49e23e55adb9119a385d8b3688db3c9024b1bf5afa3f.jpg)

![f64c9560b7403033fbd2181654e64ce2fded04d2011dcbfaa32750e27d34acff.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/f64c9560b7403033fbd2181654e64ce2fded04d2011dcbfaa32750e27d34acff.jpg)

![f6f016199fe9b533010f1dbed56c28bf31cdf23c0e67a87ba364ac200366af45.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/images/f6f016199fe9b533010f1dbed56c28bf31cdf23c0e67a87ba364ac200366af45.jpg)

### Tables

![036a8b3d0cc4160f1a6a11e852bf84c38772e712c1c37bb5e815a6617d331368.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/036a8b3d0cc4160f1a6a11e852bf84c38772e712c1c37bb5e815a6617d331368.jpg)

![0474c18794fda2fb9746e86e79288198e7fb4d753845c26ed041d8e76fd526e8.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/0474c18794fda2fb9746e86e79288198e7fb4d753845c26ed041d8e76fd526e8.jpg)

![0a438b228498e5dcdc764f4b9eb8be12ffbaeba77d7403ad35df29ae9876f420.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/0a438b228498e5dcdc764f4b9eb8be12ffbaeba77d7403ad35df29ae9876f420.jpg)

![175f42346c6d30125e7f540b860858c98e2e30ac59464136cc821953ea566265.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/175f42346c6d30125e7f540b860858c98e2e30ac59464136cc821953ea566265.jpg)

![1e914c7309ea2b1f67cac3840212f1b5edb7e65301b8ad8b3b37b6a1e604f6b8.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/1e914c7309ea2b1f67cac3840212f1b5edb7e65301b8ad8b3b37b6a1e604f6b8.jpg)

![1f3dbb0c7721ff31cece327cacb6402f90ebaf83c6c188e650bcb2046a4c3566.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/1f3dbb0c7721ff31cece327cacb6402f90ebaf83c6c188e650bcb2046a4c3566.jpg)

![21d6936fc3db41cc1e95457eab736702adb8ca47cb8cbfc91c33a68c5a52d90a.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/21d6936fc3db41cc1e95457eab736702adb8ca47cb8cbfc91c33a68c5a52d90a.jpg)

![2bb23ef63318d1130308c1a8098ec4d3670f169253166d39bcda48fd545ba364.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/2bb23ef63318d1130308c1a8098ec4d3670f169253166d39bcda48fd545ba364.jpg)

![339e48728860e433644bf8641fa42624237a40dd924b361ca59fd659ae2ef3ff.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/339e48728860e433644bf8641fa42624237a40dd924b361ca59fd659ae2ef3ff.jpg)

![5477cdc64c6f41171e726507e84017255e48347bad5af530f1c1011c3a11f283.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/5477cdc64c6f41171e726507e84017255e48347bad5af530f1c1011c3a11f283.jpg)

![68b90e71fe6a41c17476c9ffabf907f67a7a44069acfc9bea52d5514664fba71.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/68b90e71fe6a41c17476c9ffabf907f67a7a44069acfc9bea52d5514664fba71.jpg)

![696765481381c63b97f74911ec3588f49b669621f8dcbf7d01ef04b12da2da3d.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/696765481381c63b97f74911ec3588f49b669621f8dcbf7d01ef04b12da2da3d.jpg)

![832e51eb10c82bb3a99154de2c3b9f68a4c2f96f57b0a2468d0271deeded0042.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/832e51eb10c82bb3a99154de2c3b9f68a4c2f96f57b0a2468d0271deeded0042.jpg)

![86967a71ec7e4c610e61eeacf3ac753099fccc40cb044a087953ab3f869b1587.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/86967a71ec7e4c610e61eeacf3ac753099fccc40cb044a087953ab3f869b1587.jpg)

![8b04b3478995528ee36d3a30ce2ce5719e95cf3833018ec69a919f440882a1f6.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/8b04b3478995528ee36d3a30ce2ce5719e95cf3833018ec69a919f440882a1f6.jpg)

![91aae54f8c0aeffc3b535ad1b80a81fdc1afbe0d10d35723562a0159ee2eaf33.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/91aae54f8c0aeffc3b535ad1b80a81fdc1afbe0d10d35723562a0159ee2eaf33.jpg)

![94be4d5e813961f159893bfb392d375f4496bc5860bfc396ab7ab7aaf7e61cc0.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/94be4d5e813961f159893bfb392d375f4496bc5860bfc396ab7ab7aaf7e61cc0.jpg)

![9617bbbbe6089ea9b7a22b9a5d39e0f5b52f22e5183209467333b2f0a9296f76.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/9617bbbbe6089ea9b7a22b9a5d39e0f5b52f22e5183209467333b2f0a9296f76.jpg)

![b1794718ebd9ac710a1f1c97737cad994250c0834d1b175c1881eeb3d467201e.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/b1794718ebd9ac710a1f1c97737cad994250c0834d1b175c1881eeb3d467201e.jpg)

![be9c87b41ca87f845b075e3cec5db485febc3e86d67cbae8005595ac5af7da32.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/be9c87b41ca87f845b075e3cec5db485febc3e86d67cbae8005595ac5af7da32.jpg)

![c3d922facda02244c20f3267804bda67f1104cf1e54e8175957dd6ddc2b8e6c1.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/c3d922facda02244c20f3267804bda67f1104cf1e54e8175957dd6ddc2b8e6c1.jpg)

![d1fceebc5e8c7ce685851e5f71e8a56f24160f8d464ada519393bb25dc9e5aa5.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/d1fceebc5e8c7ce685851e5f71e8a56f24160f8d464ada519393bb25dc9e5aa5.jpg)

![d9de28bec5bb5c6cdb5e82203af7dacd236c230e1857ffbc8cdbd6762b3b98e0.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/d9de28bec5bb5c6cdb5e82203af7dacd236c230e1857ffbc8cdbd6762b3b98e0.jpg)

![ddf84606f7f7d9a2606c93bfb3b7f653ee033747ac95a7a151e8552e7f837ab3.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/ddf84606f7f7d9a2606c93bfb3b7f653ee033747ac95a7a151e8552e7f837ab3.jpg)

![f9d3899b0555d9dd25096951151a3d9cd0ce802b34697ed81cb48c84eaea38f0.jpg](../icml_results/4293_What%20Reward%20Structure%20Enables%20Efficient%20Sparse-Reward%20RL_%20A%20Proof-of-Concept%20with%20Policy-Aware%20Matri/tables/f9d3899b0555d9dd25096951151a3d9cd0ce802b34697ed81cb48c84eaea38f0.jpg)

<a id="paper-4251"></a>

## Infinite-Precision Autoregressive Modeling for Vector Graphics and Layouts


### Images

![0de5d6be42a7dfc745f2fb84ce202ba67469fdbf8141fc9e2d3075e38d514446.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/0de5d6be42a7dfc745f2fb84ce202ba67469fdbf8141fc9e2d3075e38d514446.jpg)

![279fcaca886707bef899f21bc967c9ffa6297b8590e49037265352c86a2a3581.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/279fcaca886707bef899f21bc967c9ffa6297b8590e49037265352c86a2a3581.jpg)

![33c4f2a65b1186a24235d0fb836a54b30ae5f166c296a9bb50675c1f2289fecb.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/33c4f2a65b1186a24235d0fb836a54b30ae5f166c296a9bb50675c1f2289fecb.jpg)

![4f5bbb2690f7b7138e06ce19fc184d2f8c5604f67f7bc107d8f378fdd03583e8.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/4f5bbb2690f7b7138e06ce19fc184d2f8c5604f67f7bc107d8f378fdd03583e8.jpg)

![60c82520ed2c2ee1a918bcc4c672f451e357b59ba415b94b7753981aeb23b2d0.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/60c82520ed2c2ee1a918bcc4c672f451e357b59ba415b94b7753981aeb23b2d0.jpg)

![61eae507f620ee387d86e9d34293c3679e9434a98e7302465e8b0cca522eafcd.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/61eae507f620ee387d86e9d34293c3679e9434a98e7302465e8b0cca522eafcd.jpg)

![85e51ffdf8d2f804863b5ca0f15169e470c6ffa06ba393d2cbd6baed080f6545.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/85e51ffdf8d2f804863b5ca0f15169e470c6ffa06ba393d2cbd6baed080f6545.jpg)

![8a02d0bacd31bf58f6b5faeabef8aa65d452c0c78866b7ccbc499ee4d1443beb.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/8a02d0bacd31bf58f6b5faeabef8aa65d452c0c78866b7ccbc499ee4d1443beb.jpg)

![a78f93e2fec108c6bb06c22e9e7d423c39625b87e6e14e7a3d2a3768e9c99d4c.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/a78f93e2fec108c6bb06c22e9e7d423c39625b87e6e14e7a3d2a3768e9c99d4c.jpg)

![ac15374c7dca34dd3a54f25e8fb16dfe6613c12f2c3f33acf078c5b2e52451ca.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/ac15374c7dca34dd3a54f25e8fb16dfe6613c12f2c3f33acf078c5b2e52451ca.jpg)

![b2b3272056b1be7583856de3d0684aaf0c2423d1f8e75a69ac9616a5f924bde1.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/b2b3272056b1be7583856de3d0684aaf0c2423d1f8e75a69ac9616a5f924bde1.jpg)

![c6d96efe1aea8825a106e4cd4ab9cfd4c9077b3d1b07cecaac84e07fc24a8722.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/c6d96efe1aea8825a106e4cd4ab9cfd4c9077b3d1b07cecaac84e07fc24a8722.jpg)

![d69a1a6190a49594ee33d16d4aa39dba18e07a3a26d44052b07fef83fee3a785.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/images/d69a1a6190a49594ee33d16d4aa39dba18e07a3a26d44052b07fef83fee3a785.jpg)

### Tables

![07e7bedecf44a96a2d3c6bc63fd03729686fce80c1e04ea0ddb1d3020452baa4.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/tables/07e7bedecf44a96a2d3c6bc63fd03729686fce80c1e04ea0ddb1d3020452baa4.jpg)

![1e9d4b38849dc2787c09a758c7af49ffd0e5fe894a236e91823bb2104b6408c3.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/tables/1e9d4b38849dc2787c09a758c7af49ffd0e5fe894a236e91823bb2104b6408c3.jpg)

![8c17b7dc1c8dada2d092a589ca5d9a329726cb17e6f0ca42ec9046d53c507d41.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/tables/8c17b7dc1c8dada2d092a589ca5d9a329726cb17e6f0ca42ec9046d53c507d41.jpg)

![8eac6824528a5d3f57cffad3dd824ba855a4e5a69d4592b69e7737faf6bb2d10.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/tables/8eac6824528a5d3f57cffad3dd824ba855a4e5a69d4592b69e7737faf6bb2d10.jpg)

![a996bca8caf3ab445ddc2c10320edeed00f678901a96ebc381b651f1dcd491da.jpg](../icml_results/4294_Infinite-Precision%20Autoregressive%20Modeling%20for%20Vector%20Graphics%20and%20Layouts/tables/a996bca8caf3ab445ddc2c10320edeed00f678901a96ebc381b651f1dcd491da.jpg)

<a id="paper-4252"></a>

## Geometry-Correct Diffusion Posterior Sampling with Denoiser-Pullback Curvature Guidance and Manifold-Aligned Damping


### Images

![0213a25638f47fd60856d64e1fe62c937633419846e168bd85afa0860b36e29f.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/0213a25638f47fd60856d64e1fe62c937633419846e168bd85afa0860b36e29f.jpg)

![0230add8c7d19012fc7f09dc179a4329624c8c864baa13d761af36e7bc998baf.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/0230add8c7d19012fc7f09dc179a4329624c8c864baa13d761af36e7bc998baf.jpg)

![317c4726ddeb8135e38f3ea656a2743f4241b180834d5e0619538146ce2588e3.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/317c4726ddeb8135e38f3ea656a2743f4241b180834d5e0619538146ce2588e3.jpg)

![32ef0fb49e33cb71e9baf9932874e3e5e54db9fd7833656bf00764c1aa8919eb.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/32ef0fb49e33cb71e9baf9932874e3e5e54db9fd7833656bf00764c1aa8919eb.jpg)

![420d233e278479ab94a59ba1762895917a39ae8bd46ca67dba118cc761a5b981.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/420d233e278479ab94a59ba1762895917a39ae8bd46ca67dba118cc761a5b981.jpg)

![605eaa9788e4d27a571f9bdd3b334d172eb7ad432bd9c81bad0ac779f26e1285.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/605eaa9788e4d27a571f9bdd3b334d172eb7ad432bd9c81bad0ac779f26e1285.jpg)

![7e6133c18c5e686c1e755f6db559d8dd2e7628a3bc64aca202af5758a3791aa8.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/7e6133c18c5e686c1e755f6db559d8dd2e7628a3bc64aca202af5758a3791aa8.jpg)

![7ed25071edaef04e1787ccfb74d28b65b6a9773dd8bb02c2d2f78d3f4acce948.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/7ed25071edaef04e1787ccfb74d28b65b6a9773dd8bb02c2d2f78d3f4acce948.jpg)

![86143aa991c7fdc639492d879616f4b675be77f9bee621a6918917eb076949b4.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/86143aa991c7fdc639492d879616f4b675be77f9bee621a6918917eb076949b4.jpg)

![bc9d5702e5c70061ed1d3f9d03e71af34b4b2fb5d962b5e0b7a5eef5c56fd551.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/bc9d5702e5c70061ed1d3f9d03e71af34b4b2fb5d962b5e0b7a5eef5c56fd551.jpg)

![be8959dc05753b78921f3a83c8ddb7e6dd0fec8035dd129969f744746464aecf.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/be8959dc05753b78921f3a83c8ddb7e6dd0fec8035dd129969f744746464aecf.jpg)

![c0054940379ccca0ef1d668c6e291aebffff7c433911e62d510bebdbc16f1c1b.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/c0054940379ccca0ef1d668c6e291aebffff7c433911e62d510bebdbc16f1c1b.jpg)

![c45c9474670d382ee984dbb7d3203a17b93181f822e9d45e242d75f42f0c1e8b.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/c45c9474670d382ee984dbb7d3203a17b93181f822e9d45e242d75f42f0c1e8b.jpg)

![dcbedf67dfe33706ca8277470d7145fbef86edf6111b21b3433321fc888c8ff4.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/dcbedf67dfe33706ca8277470d7145fbef86edf6111b21b3433321fc888c8ff4.jpg)

![eb1f17060c9d13bfa2ddc36a5e565ea28f8331dd3ddbc1b09af15964abe78908.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/images/eb1f17060c9d13bfa2ddc36a5e565ea28f8331dd3ddbc1b09af15964abe78908.jpg)

### Tables

![17fa0f746e4cad811cfe82776d975cf5c9bb01eebd4249ad09da1a707d27cb1a.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/17fa0f746e4cad811cfe82776d975cf5c9bb01eebd4249ad09da1a707d27cb1a.jpg)

![1fbdb134800c91d5fd55d1b18f41e8da5e3916a3cd65a68611913a0213c2f1df.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/1fbdb134800c91d5fd55d1b18f41e8da5e3916a3cd65a68611913a0213c2f1df.jpg)

![2f4719f9064efe8390b3675ce833b5409f225fb2cde152d52b6043c13fb928e1.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/2f4719f9064efe8390b3675ce833b5409f225fb2cde152d52b6043c13fb928e1.jpg)

![3dd18f904ce4c321dac07ec0910afbaab802e1b213617ffbcca4a5f3ec3c11cb.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/3dd18f904ce4c321dac07ec0910afbaab802e1b213617ffbcca4a5f3ec3c11cb.jpg)

![3eae1bcb6e09b50aa47a5211606cacdf7e7cb110d90b1317247640b1955f2286.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/3eae1bcb6e09b50aa47a5211606cacdf7e7cb110d90b1317247640b1955f2286.jpg)

![4e8d9959e7d6b6b565aac4ae00677db6280c1e2c5348e35506a8b321891ab01e.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/4e8d9959e7d6b6b565aac4ae00677db6280c1e2c5348e35506a8b321891ab01e.jpg)

![5a9d1ba114a74d143f8f9b6b62687f8d8b03066480b2f3e678c2ca1eaeb0ef3a.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/5a9d1ba114a74d143f8f9b6b62687f8d8b03066480b2f3e678c2ca1eaeb0ef3a.jpg)

![7417b63eb8ed5a2f968459f655f4f59422366feec7a412d5c8b85475c1e2cce2.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/7417b63eb8ed5a2f968459f655f4f59422366feec7a412d5c8b85475c1e2cce2.jpg)

![9a382378bd2ed805f995492cc708a0af396e18a83d4f5df8269aab00065a364c.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/9a382378bd2ed805f995492cc708a0af396e18a83d4f5df8269aab00065a364c.jpg)

![a9bd63ab735e2f4dea1c55bec2fe31b3ab07b6b5df1cf5dbd16d64c92ccedad2.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/a9bd63ab735e2f4dea1c55bec2fe31b3ab07b6b5df1cf5dbd16d64c92ccedad2.jpg)

![aecd7f56ebb4f784a598268455fc5c18c10e4a90369f71b07b4ce7c986dbb78d.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/aecd7f56ebb4f784a598268455fc5c18c10e4a90369f71b07b4ce7c986dbb78d.jpg)

![c5c05120200f114c73989ac6e9e31ee433996d5a1ef8dfe749b64217e7efe8d7.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/c5c05120200f114c73989ac6e9e31ee433996d5a1ef8dfe749b64217e7efe8d7.jpg)

![ccf2695a3e3077fd88302586d3dc1dc3f54261f0dbbcbf1888f23a151db739f4.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/ccf2695a3e3077fd88302586d3dc1dc3f54261f0dbbcbf1888f23a151db739f4.jpg)

![e443dd215b801de2bc5234a5d8cb3bb3a1c90cf507b95cda98a4b2ead2f9e4d6.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/e443dd215b801de2bc5234a5d8cb3bb3a1c90cf507b95cda98a4b2ead2f9e4d6.jpg)

![e504a5513dd30a50bce74a9ee5f1a21fcb769445948cb5f4e6e791106b061334.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/e504a5513dd30a50bce74a9ee5f1a21fcb769445948cb5f4e6e791106b061334.jpg)

![ec49ddf9804e60fd44525e367762d10aaf30b91ee04d4e637d42a4d0d1624c2d.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/ec49ddf9804e60fd44525e367762d10aaf30b91ee04d4e637d42a4d0d1624c2d.jpg)

![f1f1f1ade4133d3ffe88801fdb3254f26c3f2e1d4712c50b2cdadb3eb40cfaf2.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/f1f1f1ade4133d3ffe88801fdb3254f26c3f2e1d4712c50b2cdadb3eb40cfaf2.jpg)

![f86285c41718d4e0a09017d5254cbede07c65f97077f59f0210292d4ef99e174.jpg](../icml_results/4295_Geometry-Correct%20Diffusion%20Posterior%20Sampling%20with%20Denoiser-Pullback%20Curvature%20Guidance%20and%20Manifold/tables/f86285c41718d4e0a09017d5254cbede07c65f97077f59f0210292d4ef99e174.jpg)

<a id="paper-4253"></a>

## Query-Based Asymmetric Modeling with Decoupled Input–Output Rates for Speech Restoration


### Images

![1842a6e20accd4dddabaffefba58de84dbcc0c4ffc5361dbadcaf1b7e622e2fd.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/1842a6e20accd4dddabaffefba58de84dbcc0c4ffc5361dbadcaf1b7e622e2fd.jpg)

![2cbad3212d5e3fc943af26f9152a34904e972082c80b546aaf20a12a28382d7f.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/2cbad3212d5e3fc943af26f9152a34904e972082c80b546aaf20a12a28382d7f.jpg)

![2f156927fd968157c0c9ce1437e9ba28bd1acbdaa520a1d57c19617d2f8c7cba.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/2f156927fd968157c0c9ce1437e9ba28bd1acbdaa520a1d57c19617d2f8c7cba.jpg)

![3652531607b07e8d117eb79b5ed4fb25b4cbf9b04754998b2888bcf63db7a2c0.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/3652531607b07e8d117eb79b5ed4fb25b4cbf9b04754998b2888bcf63db7a2c0.jpg)

![a44606cdf273eb1aecf345f285b9ef3e217f1fbe9bc677affea82b293539097d.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/a44606cdf273eb1aecf345f285b9ef3e217f1fbe9bc677affea82b293539097d.jpg)

![d555123a0f4a72d1e55cd35e8cd0b1b737192d8c064995ad730bfb89df5fc5c3.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/d555123a0f4a72d1e55cd35e8cd0b1b737192d8c064995ad730bfb89df5fc5c3.jpg)

![e06623fd9b00d6ac9aafc2ac363b07c93d27d8796250a7b510200639bef5685b.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/e06623fd9b00d6ac9aafc2ac363b07c93d27d8796250a7b510200639bef5685b.jpg)

![fb820fafb321ca038b975cbf4a856fe8dcd87a45411d803af8d6c8c8696155ee.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/images/fb820fafb321ca038b975cbf4a856fe8dcd87a45411d803af8d6c8c8696155ee.jpg)

### Tables

![00227bb713eef9de14f3588081dcfcc973f24975370250ed7d96b156a865dbce.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/00227bb713eef9de14f3588081dcfcc973f24975370250ed7d96b156a865dbce.jpg)

![0b57093afdfca890edd7fb10d0942eb05919cdb3cac6db185395facdd2d9462e.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/0b57093afdfca890edd7fb10d0942eb05919cdb3cac6db185395facdd2d9462e.jpg)

![207f40c24d3c993a26f780509583a186d53058b9a84c0b6be531571231672101.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/207f40c24d3c993a26f780509583a186d53058b9a84c0b6be531571231672101.jpg)

![2a30ad821dd36a81ad16e658515e1ed400739c17f6b5f22efc637c179bec3c25.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/2a30ad821dd36a81ad16e658515e1ed400739c17f6b5f22efc637c179bec3c25.jpg)

![3073bf03b44fe17c5307a752d260c838b4e98b37cf577fc8c5be4d88168e1db3.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/3073bf03b44fe17c5307a752d260c838b4e98b37cf577fc8c5be4d88168e1db3.jpg)

![3f90379ee085c553c8c50f14045615706f6c53a3ba0c499fb3b51658269001cf.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/3f90379ee085c553c8c50f14045615706f6c53a3ba0c499fb3b51658269001cf.jpg)

![5011e4638bd20f3815f9c6f02fc77493010175bc2368dd7b287ea2c502a0fd5c.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/5011e4638bd20f3815f9c6f02fc77493010175bc2368dd7b287ea2c502a0fd5c.jpg)

![5f2548dc9913c9b530e1ba2f80450df197234f553680a20bd61005cf35d1c247.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/5f2548dc9913c9b530e1ba2f80450df197234f553680a20bd61005cf35d1c247.jpg)

![63b216805e1dbd952881a908080d97e1b7a40e0e7cfe4970f3209bbf13a00351.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/63b216805e1dbd952881a908080d97e1b7a40e0e7cfe4970f3209bbf13a00351.jpg)

![7191d6bd6acc83e07d540c6794130f8cf619ce8103c641858f98498aa5a27697.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/7191d6bd6acc83e07d540c6794130f8cf619ce8103c641858f98498aa5a27697.jpg)

![7299bcf058b06dfbbb50975fa50b05683a31fcc38fc5b841b1f916d5e69e97e2.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/7299bcf058b06dfbbb50975fa50b05683a31fcc38fc5b841b1f916d5e69e97e2.jpg)

![7c4cb5ba0c966623cb65db5acd7c832b2254426b85a047a1e2b1c390a8f066dc.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/7c4cb5ba0c966623cb65db5acd7c832b2254426b85a047a1e2b1c390a8f066dc.jpg)

![801d9b19fe5eec4a26221302877a7e9a1b288c6129c594219b29fa6d1b73c5a5.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/801d9b19fe5eec4a26221302877a7e9a1b288c6129c594219b29fa6d1b73c5a5.jpg)

![8cd9dd8d9ec5b000098d1d6a2625cf2cda545c0101bb3b169bc10034addab3dd.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/8cd9dd8d9ec5b000098d1d6a2625cf2cda545c0101bb3b169bc10034addab3dd.jpg)

![92b8ad221085dbe7fb1b0fda8501c4f0a3e8dd0bb1d044c0401226a3b1bb1d1c.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/92b8ad221085dbe7fb1b0fda8501c4f0a3e8dd0bb1d044c0401226a3b1bb1d1c.jpg)

![a4ba26e4dba9ef5a5573387eef38a975cf87d0c8e744daaed70608c0a980ef84.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/a4ba26e4dba9ef5a5573387eef38a975cf87d0c8e744daaed70608c0a980ef84.jpg)

![a506081dae3a1eeba846699f159137053877e4af361ad69f664457bf484a83c2.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/a506081dae3a1eeba846699f159137053877e4af361ad69f664457bf484a83c2.jpg)

![a82a046a2eab2657d50bc685f273f4121eb966bd2da4dcce684af3fa125d98a0.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/a82a046a2eab2657d50bc685f273f4121eb966bd2da4dcce684af3fa125d98a0.jpg)

![acae1c8bd44893c9078dfd9fdcaa276cf14f6cd1464ab6a6895a604280979f6d.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/acae1c8bd44893c9078dfd9fdcaa276cf14f6cd1464ab6a6895a604280979f6d.jpg)

![b24d2fe559f653e18a97652af252c136644119b2323c762bda8588e922730943.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/b24d2fe559f653e18a97652af252c136644119b2323c762bda8588e922730943.jpg)

![d51eb97b5b555b361c71cdc953e11767df5f4330c74fedd247e9554c46536c20.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/d51eb97b5b555b361c71cdc953e11767df5f4330c74fedd247e9554c46536c20.jpg)

![d67f726eed49537099671bd36fb310ed5c77800195e53d0565742de81c8f6cec.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/d67f726eed49537099671bd36fb310ed5c77800195e53d0565742de81c8f6cec.jpg)

![e05538e97a14f5f799f96d5111f3a00cfb6c4f75f0262f9ec249171a92cdf59e.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/e05538e97a14f5f799f96d5111f3a00cfb6c4f75f0262f9ec249171a92cdf59e.jpg)

![e69c199b6e9bd2bb3c56f63480d4bf4679a2a44b687170f2d94ce452c57f7ac8.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/e69c199b6e9bd2bb3c56f63480d4bf4679a2a44b687170f2d94ce452c57f7ac8.jpg)

![f52b3262d0a6f1f31a3ad0d69e8ff39babdf2030b2aed22e0a8b19d83f8f0a3d.jpg](../icml_results/4296_Query-Based%20Asymmetric%20Modeling%20with%20Decoupled%20Input%E2%80%93Output%20Rates%20for%20Speech%20Restoration/tables/f52b3262d0a6f1f31a3ad0d69e8ff39babdf2030b2aed22e0a8b19d83f8f0a3d.jpg)

<a id="paper-4254"></a>

## Efficient Generative Modeling beyond Memoryless Diffusion via Adjoint Schrödinger Bridge Matching


### Images

![4b4bdadb75f373633971d5df0c77cd4ee387c73af940899bfba8929acc8401c4.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/4b4bdadb75f373633971d5df0c77cd4ee387c73af940899bfba8929acc8401c4.jpg)

![5c731cff31a06d3dc928457354c581e7357d25d04a7ccff5bddb1639e893391c.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/5c731cff31a06d3dc928457354c581e7357d25d04a7ccff5bddb1639e893391c.jpg)

![7007dd14eb0d471dfbdfd83f7cf7c2b7cd7695ec246e0a816f0399281b423f20.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/7007dd14eb0d471dfbdfd83f7cf7c2b7cd7695ec246e0a816f0399281b423f20.jpg)

![72ccfb5ee10162bfd872b38146f865615e60b0a813445f70528d38d8a0c08c80.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/72ccfb5ee10162bfd872b38146f865615e60b0a813445f70528d38d8a0c08c80.jpg)

![bdc5a9d84ecae4bb7c76695717c828162c836a6f959e334b9a51b49524c40fe1.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/bdc5a9d84ecae4bb7c76695717c828162c836a6f959e334b9a51b49524c40fe1.jpg)

![d9641bda7e8880337a390dcedde679f59a6b7eb65f551e0c968ff07a934326f3.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/d9641bda7e8880337a390dcedde679f59a6b7eb65f551e0c968ff07a934326f3.jpg)

![de41910887a01cf05cf8f09d561b5426d1c2e9a1594fbf0375bc577610b8344d.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/de41910887a01cf05cf8f09d561b5426d1c2e9a1594fbf0375bc577610b8344d.jpg)

![e81cff90d7b3aa5c1d26efcdc5487064f53972af307541fb8862dcdacd5ad5aa.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/e81cff90d7b3aa5c1d26efcdc5487064f53972af307541fb8862dcdacd5ad5aa.jpg)

![ee443590a5e8f85d165f782b9b2c683e9cf2ff7c45507037108c9aac13d60810.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/ee443590a5e8f85d165f782b9b2c683e9cf2ff7c45507037108c9aac13d60810.jpg)

![f675cc4a52791c4943795813ceee97f26f087ea70ca91d83c09e1f3eca74d5d2.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/images/f675cc4a52791c4943795813ceee97f26f087ea70ca91d83c09e1f3eca74d5d2.jpg)

### Tables

![009a4c9fc641bd9e8e47d1a4a4d695d3ae755cb81a6d5774f3dcf0b6f514ea15.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/tables/009a4c9fc641bd9e8e47d1a4a4d695d3ae755cb81a6d5774f3dcf0b6f514ea15.jpg)

![39624ac747fff5b83b1f1d23cfb9cdde2d7eaeb6dfd5be135a1dd283f9bcb4e2.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/tables/39624ac747fff5b83b1f1d23cfb9cdde2d7eaeb6dfd5be135a1dd283f9bcb4e2.jpg)

![666ecd5ca232d35e293a34edac88cc308365d1161d437a04338aa13a3d8c64d7.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/tables/666ecd5ca232d35e293a34edac88cc308365d1161d437a04338aa13a3d8c64d7.jpg)

![716119c6288edc6848d746b7c7c1fc7e073387e727169123d39a3fb1ab79278b.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/tables/716119c6288edc6848d746b7c7c1fc7e073387e727169123d39a3fb1ab79278b.jpg)

![9aab99895ac6091c93efd487bb8802ef1440770c71f18006893fbbac6c991e0b.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/tables/9aab99895ac6091c93efd487bb8802ef1440770c71f18006893fbbac6c991e0b.jpg)

![d065d56dfed670b7c8a491911bfb2110f71007452352453cec5d49e6fda5f5e2.jpg](../icml_results/4297_Efficient%20Generative%20Modeling%20beyond%20Memoryless%20Diffusion%20via%20Adjoint%20Schr%C3%B6dinger%20Bridge%20Matching/tables/d065d56dfed670b7c8a491911bfb2110f71007452352453cec5d49e6fda5f5e2.jpg)

<a id="paper-4255"></a>

## Parsimonious Learning-Augmented Online Metric Matching


### Images

![0936458dc29596f8fbdc26975597840e07416582ae2e947295f4dea3a740f19f.jpg](../icml_results/4298_Parsimonious%20Learning-Augmented%20Online%20Metric%20Matching/images/0936458dc29596f8fbdc26975597840e07416582ae2e947295f4dea3a740f19f.jpg)

![514830918d3f94a3b5408b00831692e82112e4828c562d6c50ace390a379a9cb.jpg](../icml_results/4298_Parsimonious%20Learning-Augmented%20Online%20Metric%20Matching/images/514830918d3f94a3b5408b00831692e82112e4828c562d6c50ace390a379a9cb.jpg)

### Tables

![45a9c625ee46d000380ec74e3cd3df6b8636dc879af50567cf18a54bde04a4d0.jpg](../icml_results/4298_Parsimonious%20Learning-Augmented%20Online%20Metric%20Matching/tables/45a9c625ee46d000380ec74e3cd3df6b8636dc879af50567cf18a54bde04a4d0.jpg)

![7d69ac53035ce98535fa1f0f951ae48dfb064ec09093368079c8644c8a618b58.jpg](../icml_results/4298_Parsimonious%20Learning-Augmented%20Online%20Metric%20Matching/tables/7d69ac53035ce98535fa1f0f951ae48dfb064ec09093368079c8644c8a618b58.jpg)

<a id="paper-4256"></a>

## Overclocking Electrostatic Generative Models


### Images

![084a185073393e8a866bd3d3ccf6e2dcdc06de979966db1dc3588c0856ba4e22.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/084a185073393e8a866bd3d3ccf6e2dcdc06de979966db1dc3588c0856ba4e22.jpg)

![1d51f19d828718426a44b9165fc0a48c1e977586f9c765790d8d1401006c84c4.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/1d51f19d828718426a44b9165fc0a48c1e977586f9c765790d8d1401006c84c4.jpg)

![1dacb9c80b4f711df7d392e3fa2820f48cc5f00dddbc0e011d4c16a7d1c93147.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/1dacb9c80b4f711df7d392e3fa2820f48cc5f00dddbc0e011d4c16a7d1c93147.jpg)

![220de14ed94de1b4ed54cd3a55a54f59d77e0f18b1b2c52ae44ea6779f1f7858.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/220de14ed94de1b4ed54cd3a55a54f59d77e0f18b1b2c52ae44ea6779f1f7858.jpg)

![2628f591534430635d8563c5a8ca0ec2eb1b1cf4f9dbf881996b604d101ea922.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/2628f591534430635d8563c5a8ca0ec2eb1b1cf4f9dbf881996b604d101ea922.jpg)

![2903f619dab7122fc91bdf9a386b92b4da75b6c9bfd05dc6bae38a39250781a3.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/2903f619dab7122fc91bdf9a386b92b4da75b6c9bfd05dc6bae38a39250781a3.jpg)

![2e7543fb36f5c0535365d9d1383db364738814a340adc11352f570d546e6b369.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/2e7543fb36f5c0535365d9d1383db364738814a340adc11352f570d546e6b369.jpg)

![391fca39ce0d801d283cbcba23af968e7350ddd9b07fb679263bc79e124f79a2.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/391fca39ce0d801d283cbcba23af968e7350ddd9b07fb679263bc79e124f79a2.jpg)

![4b18be025845852e755272aeb1d203d560be2a06ccb41084a892b5d72a324e33.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/4b18be025845852e755272aeb1d203d560be2a06ccb41084a892b5d72a324e33.jpg)

![5f639fc0c03580f5f5bbcfda03f831692088ee3cb40634c43ffe46f245432776.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/5f639fc0c03580f5f5bbcfda03f831692088ee3cb40634c43ffe46f245432776.jpg)

![621180249de37b869206c76b209b05f66093ad9acec14c70b523f4034137e460.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/621180249de37b869206c76b209b05f66093ad9acec14c70b523f4034137e460.jpg)

![6a1756456fa190aad753232cac4d0a79da17d47a2a3fc19c77807b9460831c22.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/6a1756456fa190aad753232cac4d0a79da17d47a2a3fc19c77807b9460831c22.jpg)

![76846c0d7f34ea0e79aa0f758555779ed45863eb90d19cdc3471b89971514b40.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/76846c0d7f34ea0e79aa0f758555779ed45863eb90d19cdc3471b89971514b40.jpg)

![771dc8158ff7ca11b57e64f92697659815d9e20738f6bb048078f219f21172d6.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/771dc8158ff7ca11b57e64f92697659815d9e20738f6bb048078f219f21172d6.jpg)

![77b6006c7e53c925d13fa535a9c96c1705739921036696045e101b4aade270c3.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/77b6006c7e53c925d13fa535a9c96c1705739921036696045e101b4aade270c3.jpg)

![9285bc6547d79f7bc43048142ded6de8cca8f93aa04c7e06f6c5a39ca9bca944.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/9285bc6547d79f7bc43048142ded6de8cca8f93aa04c7e06f6c5a39ca9bca944.jpg)

![928a32157056011605ce439a10a782e9701289793b6f5f303d62f1676c284ed7.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/928a32157056011605ce439a10a782e9701289793b6f5f303d62f1676c284ed7.jpg)

![978282b5a0c1dc836abdf58bb004bd4cb37a166f9fd7aaae927f6e757d512573.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/978282b5a0c1dc836abdf58bb004bd4cb37a166f9fd7aaae927f6e757d512573.jpg)

![adefede822199978349d3d45b918cdb5c9e5554204f164c71f89e3d5439c9af9.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/adefede822199978349d3d45b918cdb5c9e5554204f164c71f89e3d5439c9af9.jpg)

![b69b7382720f255267908a14da03d7e4fcaacb77c06d379025d84554de37668e.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/b69b7382720f255267908a14da03d7e4fcaacb77c06d379025d84554de37668e.jpg)

![b8e5ddf7dcc2fc7fcf5475322990ff8d67e4e2740b7f91596a3664f2ebf2a8a9.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/b8e5ddf7dcc2fc7fcf5475322990ff8d67e4e2740b7f91596a3664f2ebf2a8a9.jpg)

![be9171b6199c75b5cfbd4c9a9b6e6100f18f569e627fe4c420added9a1d21786.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/be9171b6199c75b5cfbd4c9a9b6e6100f18f569e627fe4c420added9a1d21786.jpg)

![bf73cb189205b1614d9773f8914e293f84104720465328f6a723447e50d4ef77.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/bf73cb189205b1614d9773f8914e293f84104720465328f6a723447e50d4ef77.jpg)

![cb5cfce9042e4656af95ecdd460671327fa5c67de5c2e1f709937862da958a22.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/cb5cfce9042e4656af95ecdd460671327fa5c67de5c2e1f709937862da958a22.jpg)

![e46d5a9dd1d31be702e0e70e6e3a9c1e3a9781a8215dd824bf203b2b756126c4.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/e46d5a9dd1d31be702e0e70e6e3a9c1e3a9781a8215dd824bf203b2b756126c4.jpg)

![eb5b389ad5f4c6114fc7b8df1c71581c260701ad0bf326855d8dcb6f757aff86.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/eb5b389ad5f4c6114fc7b8df1c71581c260701ad0bf326855d8dcb6f757aff86.jpg)

![f577f7e4cf99e3f3ed2e4e317fb50644ad17063524e95547f264bb91d1868fe0.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/images/f577f7e4cf99e3f3ed2e4e317fb50644ad17063524e95547f264bb91d1868fe0.jpg)

### Tables

![01d905069b9621da35449c0d359d2c85516a447186d4fb258b344aabbff26371.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/tables/01d905069b9621da35449c0d359d2c85516a447186d4fb258b344aabbff26371.jpg)

![330f51f613c10251e4f13b5a2d3692159976949cd2a9f7dcfb071f5d65f746ce.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/tables/330f51f613c10251e4f13b5a2d3692159976949cd2a9f7dcfb071f5d65f746ce.jpg)

![92e5eb5b5c94ce6ad99fda87b2654a871ab9d41d3ec3b1a4b75f6c3a9dee6c2e.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/tables/92e5eb5b5c94ce6ad99fda87b2654a871ab9d41d3ec3b1a4b75f6c3a9dee6c2e.jpg)

![e1dc7ffd0995bbd98f739e7fbd2651514694548def6c6b8267d2035201415a28.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/tables/e1dc7ffd0995bbd98f739e7fbd2651514694548def6c6b8267d2035201415a28.jpg)

![eaf4089ca26c2d86854d825afc4c6b50fafcbfeaa220b5e8f89944343f9e67dd.jpg](../icml_results/4299_Overclocking%20Electrostatic%20Generative%20Models/tables/eaf4089ca26c2d86854d825afc4c6b50fafcbfeaa220b5e8f89944343f9e67dd.jpg)

<a id="paper-4257"></a>

## Responsible Text-to-Image Diffusion: Interpretable and Linearly Controllable Semantics for Fair and Safe Generation

### Images

![0cea3b0a44eb2d5d131152dc669dc091847d8b8737d023e1538bedbd1efabafd.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/0cea3b0a44eb2d5d131152dc669dc091847d8b8737d023e1538bedbd1efabafd.jpg)

![0d3f560c5eb87d60d3d901732448bdf72c438d78caf19519839ad37e4ead2741.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/0d3f560c5eb87d60d3d901732448bdf72c438d78caf19519839ad37e4ead2741.jpg)

![106f637a270bfaed70a7436ce3edf143f7bc7e9057679041e65468d6b148e125.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/106f637a270bfaed70a7436ce3edf143f7bc7e9057679041e65468d6b148e125.jpg)

![14006cf82cb79942e29eaed74ca28a3411291fe13571bcaf6771385145e36f3c.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/14006cf82cb79942e29eaed74ca28a3411291fe13571bcaf6771385145e36f3c.jpg)

![21de9e6b616d9fcbc8cd6bfe4a65d591f08ca38dae7f5038e66f3607cbb96cd3.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/21de9e6b616d9fcbc8cd6bfe4a65d591f08ca38dae7f5038e66f3607cbb96cd3.jpg)

![2b1cdb06c853ad5c5cc83d0bc7e0840222e730fa60c5669e63226ef7157dfaf4.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/2b1cdb06c853ad5c5cc83d0bc7e0840222e730fa60c5669e63226ef7157dfaf4.jpg)

![53ffb9b72107a47c46cd78093f934d1ae9d35522e9bd518de6492485982529d5.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/53ffb9b72107a47c46cd78093f934d1ae9d35522e9bd518de6492485982529d5.jpg)

![84fd90eb4fd5e151ec2663a83e4940138282dd518f041fc208a57eecc1405abb.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/84fd90eb4fd5e151ec2663a83e4940138282dd518f041fc208a57eecc1405abb.jpg)

![b967780d945fb883f0b5e3506424a64904012962f29a1c1bc4e47fe050723466.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/b967780d945fb883f0b5e3506424a64904012962f29a1c1bc4e47fe050723466.jpg)

![b99398431381aca6c38897bfd5cf3ef1bc50001d9ad1bce53eda5790ea602a15.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/b99398431381aca6c38897bfd5cf3ef1bc50001d9ad1bce53eda5790ea602a15.jpg)

![c346a838af7b86dc4223f8ac79d9a19fa98b37ba4db66ddfe724e30ce1603277.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/c346a838af7b86dc4223f8ac79d9a19fa98b37ba4db66ddfe724e30ce1603277.jpg)

![d18cec520324a8e4213dfa50922a6dbda7074ba18f13f1cafab748a24f0a9c66.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/d18cec520324a8e4213dfa50922a6dbda7074ba18f13f1cafab748a24f0a9c66.jpg)

![de6bb9c10ac3d9822ed35137b2945064780874f5d52669f95afe9d304e79cb05.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/de6bb9c10ac3d9822ed35137b2945064780874f5d52669f95afe9d304e79cb05.jpg)

![f743c3ec7a659fa057e0797a818d9c19ee885398fefa7d3e70c56728f2cb240a.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/f743c3ec7a659fa057e0797a818d9c19ee885398fefa7d3e70c56728f2cb240a.jpg)

![fcec80744a22cb31b4dd7a0dc9a19d36072da6d8d36616d07e7fc335086f22c1.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/images/fcec80744a22cb31b4dd7a0dc9a19d36072da6d8d36616d07e7fc335086f22c1.jpg)

### Tables

![3cdb45d98fce117e0df1df242fb26986ff12f4f26b0e90f15d181d96f9d47df4.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/3cdb45d98fce117e0df1df242fb26986ff12f4f26b0e90f15d181d96f9d47df4.jpg)

![45c48d84cfd2618ff2455c33ce19a01db027c4232a57b85620043bfd9d78d3ee.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/45c48d84cfd2618ff2455c33ce19a01db027c4232a57b85620043bfd9d78d3ee.jpg)

![49284e161b784d8fe3faab761d20c7fc853fe4ba6c5144628015bf1ef9b4be58.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/49284e161b784d8fe3faab761d20c7fc853fe4ba6c5144628015bf1ef9b4be58.jpg)

![5086adcd66b8fbd620777f3d7c99375d8711036499fac342d82e523644f5f721.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/5086adcd66b8fbd620777f3d7c99375d8711036499fac342d82e523644f5f721.jpg)

![5506bf6ddef1a752f4de1af1d58601b30e9ded0124ec9aaebf63b1c5082fdd47.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/5506bf6ddef1a752f4de1af1d58601b30e9ded0124ec9aaebf63b1c5082fdd47.jpg)

![69cbb45e462d17f66545409c673ecfb5304733d971739d2a37e878fe37aeb325.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/69cbb45e462d17f66545409c673ecfb5304733d971739d2a37e878fe37aeb325.jpg)

![7f9ae42269e16f7bd4a353ecac751420d3a08d1bfa7da6bab6742c81e2e2c548.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/7f9ae42269e16f7bd4a353ecac751420d3a08d1bfa7da6bab6742c81e2e2c548.jpg)

![89ac3fb83504eb91b51df945091fdd440f9de38bf091dc5ae0d6ebd94e38753a.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/89ac3fb83504eb91b51df945091fdd440f9de38bf091dc5ae0d6ebd94e38753a.jpg)

![941025c795c52483c100e2d2d8a455243238724aa39bf4e870d9332c4eea8b4f.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/941025c795c52483c100e2d2d8a455243238724aa39bf4e870d9332c4eea8b4f.jpg)

![afa29f2117594022db3620a3c252fa188ada6e661edee49214ba95ff77d9099f.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/afa29f2117594022db3620a3c252fa188ada6e661edee49214ba95ff77d9099f.jpg)

![d72aff98c3d74830b1e75eb3ac3773bd2f34b507928a4a12e1a4174edb78ab7e.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/d72aff98c3d74830b1e75eb3ac3773bd2f34b507928a4a12e1a4174edb78ab7e.jpg)

![e6425815bec40b66cd82f06385d0da489b793107b38bccd6d8d7d708638e4b2e.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/e6425815bec40b66cd82f06385d0da489b793107b38bccd6d8d7d708638e4b2e.jpg)

![ea6025b69d6789b71ba271238fc932823fd9a69c3556e747ac5568828d9daa5f.jpg](../icml_results/4300_Responsible%20Text-to-Image%20Diffusion_%20Interpretable%20and%20Linearly%20Controllable%20Semantics%20for%20Fair%20and%20/tables/ea6025b69d6789b71ba271238fc932823fd9a69c3556e747ac5568828d9daa5f.jpg)
