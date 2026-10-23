# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 22 of 197

## 目录 (Table of Contents)

1. [Interactive Person Retrieval via Multi-Turn Multimodal Conversation](#paper-0694)
2. [D-ARL: A Distribution-Matched Asynchronous Reinforcement Learning Framework for Language Reasoning](#paper-0695)
3. [Diving into Kronecker Adapters: Component Design Matters](#paper-0696)
4. [Optimizing Few-Step Generation with Adaptive Matching Distillation](#paper-0697)
5. [Adaptive Preconditioners Trigger Loss Spikes in Adam](#paper-0698)
6. [Compressed Sensing for Capability Localization in Large Language Models](#paper-0699)
7. [Supervised Guidance Training for Infinite-Dimensional Diffusion Models](#paper-0700)
8. [Hair-Trigger Alignment: Black-Box Evaluation Cannot Guarantee Post-Update Alignment](#paper-0701)
9. [Collaborative Threshold Watermarking](#paper-0702)
10. [miniF2F-Dafny: LLM-Guided Mathematical Theorem Proving via Auto-Active Verification](#paper-0703)
11. [Fixed Budget is No Harder Than Fixed Confidence in Best-Arm Identification up to Logarithmic Factors](#paper-0704)
12. [Dependence-Aware Label Aggregation for LLM-as-a-Judge via Ising Models](#paper-0705)
13. [Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](#paper-0706)
14. [Regret-Based Federated Causal Discovery with Unknown Interventions](#paper-0707)
15. [Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](#paper-0708)
16. [Quantifying LLM Attention-Head Stability: Implications for Circuit Universality](#paper-0709)
17. [Modeling Temporal scRNA-seq Data with Latent Gaussian Process and Optimal Transport](#paper-0710)
18. [Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making](#paper-0711)
19. [Reading Between the Tokens: Improving Preference Predictions through Mechanistic Forecasting](#paper-0712)
20. [Don’t Walk the Line: Boundary Guidance for Filtered Generation](#paper-0713)
21. [A Machine-Learned Comorbidity Index](#paper-0714)
22. [DOT-MoE: Differentiable Optimal Transport for MoEfication](#paper-0715)
23. [XRPO: Pushing the Limits of GRPO with Targeted Exploration and Exploitation](#paper-0716)
24. [EPSVec: Efficient and Private Synthetic Data Generation via Dataset Vectors](#paper-0717)
25. [RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation](#paper-0718)
26. [Local-Minima-Preserving Polynomial Relaxation of Ising Problems](#paper-0719)
27. [Efficient Continuous-Depth Modeling with GRU Equivalents](#paper-0720)
28. [Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules](#paper-0721)
29. [Adversarially Robust Approximate Furthest Neighbor](#paper-0722)
30. [Improved Dynamic Algorithm for Non-monotone Submodular Maximization under Cardinality Constraint](#paper-0723)
31. [Conformal C2ST: Turning weak classifiers into strong two-sample tests](#paper-0724)
32. [Learning Junta Distributions, Quantum Junta States, and QAC$^0$ Circuits](#paper-0725)
33. [Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](#paper-0726)

---


<a id="paper-0694"></a>

## Interactive Person Retrieval via Multi-Turn Multimodal Conversation

### Images

![77bd1f7f0cfe8709dce6a065c998bf0d394b3cdef56776642f2eabb44cbecae1.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/images/77bd1f7f0cfe8709dce6a065c998bf0d394b3cdef56776642f2eabb44cbecae1.jpg)

![dfab6f22a7d239114fa11d2c5db435979aade7a697b6da0e78444e2cf81ac8d4.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/images/dfab6f22a7d239114fa11d2c5db435979aade7a697b6da0e78444e2cf81ac8d4.jpg)

### Tables

![3ca8202da88850a0995c3045db96f353c2de24fba36186bdb1fa9721e4c18893.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/3ca8202da88850a0995c3045db96f353c2de24fba36186bdb1fa9721e4c18893.jpg)

![5697420bc90f4de56478d01d9c622c3ee3cc7d314f55699cccb931f248dc39b7.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/5697420bc90f4de56478d01d9c622c3ee3cc7d314f55699cccb931f248dc39b7.jpg)

![679a9f928e03d6a53e0139ec0f965502f1b6bc5941cf2f2f99eb2488f167dada.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/679a9f928e03d6a53e0139ec0f965502f1b6bc5941cf2f2f99eb2488f167dada.jpg)

![6887a50f1d86814cf2e953af9c8dec32ca2e341725e5675bee8a97a0f0e6c4c5.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/6887a50f1d86814cf2e953af9c8dec32ca2e341725e5675bee8a97a0f0e6c4c5.jpg)

![6ee4df014ade12ed220d3bf9a4e6fee44e1ee575e61ecd768f000a86951ccca4.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/6ee4df014ade12ed220d3bf9a4e6fee44e1ee575e61ecd768f000a86951ccca4.jpg)

![712a8195694bc360a3522532079410a50328708e5220cb261c2cc5a04befccf8.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/712a8195694bc360a3522532079410a50328708e5220cb261c2cc5a04befccf8.jpg)

![85c4e978973bd6af4d0381b0c54f4c3a07895ef7ea4701aff28b52cba4a5fd72.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/85c4e978973bd6af4d0381b0c54f4c3a07895ef7ea4701aff28b52cba4a5fd72.jpg)

![8cac30fd3472bcc2031162bbd052b68fe0d51131452c66ef954a8438f9045b17.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/8cac30fd3472bcc2031162bbd052b68fe0d51131452c66ef954a8438f9045b17.jpg)

![e8d303e43e6385b6f8ef6b7c368715525bb421c35df0e51172d35b14279f9307.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/e8d303e43e6385b6f8ef6b7c368715525bb421c35df0e51172d35b14279f9307.jpg)

![f4317b61f3ea3de2bd2b6835c8cc82067cd197bcedcde7aa4e728e0062b4efe1.jpg](../icml_results/703_GI-GCN_%20Global%20Interacted%20Graph%20Convolutional%20Networks%20via%20Dominant%20Sets%20for%20Graph%20Classification/tables/f4317b61f3ea3de2bd2b6835c8cc82067cd197bcedcde7aa4e728e0062b4efe1.jpg)

## Interactive Person Retrieval via Multi-Turn Multimodal Conversation


### Images

![1e6dfd8ab8b576d475b34dc0729d507ecda6ded57a45e5f69ef71d2da7d7e5fd.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/1e6dfd8ab8b576d475b34dc0729d507ecda6ded57a45e5f69ef71d2da7d7e5fd.jpg)

![4f3d590ab611613aa4dc71e2e175b86c3a262f15f56d5367525ddde75918d6dd.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/4f3d590ab611613aa4dc71e2e175b86c3a262f15f56d5367525ddde75918d6dd.jpg)

![8514790ebed27c8f9b0bd0adee1a6d7369b80579032fcf72525c0b955d1fd56a.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/8514790ebed27c8f9b0bd0adee1a6d7369b80579032fcf72525c0b955d1fd56a.jpg)

![baacd88768e1960b2b87165dc5dc96abace1a1694d5baa374430817a5d48f821.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/baacd88768e1960b2b87165dc5dc96abace1a1694d5baa374430817a5d48f821.jpg)

![be4b99b807354062a32f2d1ac7292fb5f752dd6572ae18e2e9477430dd33af77.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/be4b99b807354062a32f2d1ac7292fb5f752dd6572ae18e2e9477430dd33af77.jpg)

![d16d8e4f0d08cda5b2cbddf5b4df17ddee6f772a7353e6705169b89aa1ccb227.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/d16d8e4f0d08cda5b2cbddf5b4df17ddee6f772a7353e6705169b89aa1ccb227.jpg)

![d1741e23e71b43d8057a0916789d0a155a0646f13291484119613cc9ab0fd420.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/d1741e23e71b43d8057a0916789d0a155a0646f13291484119613cc9ab0fd420.jpg)

![fc6e5819bdb319563c32de449d7b190416e09eafe2a4d06db9eb888e7c1a49d9.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/images/fc6e5819bdb319563c32de449d7b190416e09eafe2a4d06db9eb888e7c1a49d9.jpg)

### Tables

![14f72b0020ebc6c4befaab2c5e8495e9c51e82d182aee60451ff22f335c7e150.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/14f72b0020ebc6c4befaab2c5e8495e9c51e82d182aee60451ff22f335c7e150.jpg)

![2d0655a6cad7c9f853ba1329bd079f0733a915a921e8c6090da9665217b670e8.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/2d0655a6cad7c9f853ba1329bd079f0733a915a921e8c6090da9665217b670e8.jpg)

![8179f6a319e4c0df4d571f121501970dde76a7833c19b137cbb99520819ba9b9.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/8179f6a319e4c0df4d571f121501970dde76a7833c19b137cbb99520819ba9b9.jpg)

![83b37350bf37df3395bf137b7c9687606c3f26ca054c6c751f30120458b0cd1c.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/83b37350bf37df3395bf137b7c9687606c3f26ca054c6c751f30120458b0cd1c.jpg)

![897fbc08864db82c83c3d5e547ee5e3f65193b420e47a4ee79b432608e260dab.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/897fbc08864db82c83c3d5e547ee5e3f65193b420e47a4ee79b432608e260dab.jpg)

![91fca420a850aac797e423030e7e041f85102831f845f75814134b59ab3ad110.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/91fca420a850aac797e423030e7e041f85102831f845f75814134b59ab3ad110.jpg)

![b30373265703b05843e7a287aea3e3de5f0be284da4d3894871803ef365acb3b.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/b30373265703b05843e7a287aea3e3de5f0be284da4d3894871803ef365acb3b.jpg)

![bcc6a8a0ea9f4c650f3d1d7a969b224341be6374b0e91fd31b7c8242cd623cf4.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/bcc6a8a0ea9f4c650f3d1d7a969b224341be6374b0e91fd31b7c8242cd623cf4.jpg)

![f3de64f6c6d6e5e987dddb591ffe60e0e51b87c12860e8838dc85770ff4d897b.jpg](../icml_results/704_Interactive%20Person%20Retrieval%20via%20Multi-Turn%20Multimodal%20Conversation/tables/f3de64f6c6d6e5e987dddb591ffe60e0e51b87c12860e8838dc85770ff4d897b.jpg)

<a id="paper-0695"></a>

## D-ARL: A Distribution-Matched Asynchronous Reinforcement Learning Framework for Language Reasoning


### Images

![0d42a93abf2f50f84dce2e1b4cb943b3d04caf6ba878de8f3453ed99d590b1c1.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/0d42a93abf2f50f84dce2e1b4cb943b3d04caf6ba878de8f3453ed99d590b1c1.jpg)

![0d7adc2a40b450326bbce143865c31b88023c72e5689fd4cf50b8ad90efd4a5d.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/0d7adc2a40b450326bbce143865c31b88023c72e5689fd4cf50b8ad90efd4a5d.jpg)

![15807a7f0e13aca8547983aaefa8e402cea8c54fff963d3cb71e4a8406cbf680.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/15807a7f0e13aca8547983aaefa8e402cea8c54fff963d3cb71e4a8406cbf680.jpg)

![364153924927f4eb2e8a05d0b7b14000a08e960d8659c178fa714adb42d1a6aa.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/364153924927f4eb2e8a05d0b7b14000a08e960d8659c178fa714adb42d1a6aa.jpg)

![53ab40dc1f1fb8235c080d1922b1dd6c009e0ed2c92ca04dff8f69621cfd69ba.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/53ab40dc1f1fb8235c080d1922b1dd6c009e0ed2c92ca04dff8f69621cfd69ba.jpg)

![5d957af6766780c41ed74a7033a5b40bc1e7bff338099dc83ebcb686391d2a0d.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/5d957af6766780c41ed74a7033a5b40bc1e7bff338099dc83ebcb686391d2a0d.jpg)

![616c3af7d5f6e8a63ba50bf0753d398461835b94430346b4cde9d9152eacb90d.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/616c3af7d5f6e8a63ba50bf0753d398461835b94430346b4cde9d9152eacb90d.jpg)

![97184422961e732f1540287fe856efd08227b652ac0fe7b9e486b7de9b604d78.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/97184422961e732f1540287fe856efd08227b652ac0fe7b9e486b7de9b604d78.jpg)

![cb3dced728712978b05b53ffcc1adfa9d846e32c6657e8f9721aa1b9e4f22ffb.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/cb3dced728712978b05b53ffcc1adfa9d846e32c6657e8f9721aa1b9e4f22ffb.jpg)

![f1e92ecfb9f30fbf23f78d7208250f4439d0ab477e0e3b4c89b10bd3320deb6f.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/images/f1e92ecfb9f30fbf23f78d7208250f4439d0ab477e0e3b4c89b10bd3320deb6f.jpg)

### Tables

![12f8a0f8755b09070bad07fe9b7d719fead786bf5c9e355a963b2d2fd89f5720.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/tables/12f8a0f8755b09070bad07fe9b7d719fead786bf5c9e355a963b2d2fd89f5720.jpg)

![3149c6bcf12e2d0a9209c992d024a1fe2c3bf805973158c4b9962f1a464a0706.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/tables/3149c6bcf12e2d0a9209c992d024a1fe2c3bf805973158c4b9962f1a464a0706.jpg)

![3616270f9318c64d2705f27bc22132aa36da340840bf34ccfbd81ffa3ea0a080.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/tables/3616270f9318c64d2705f27bc22132aa36da340840bf34ccfbd81ffa3ea0a080.jpg)

![421bc948a42b4e6643c80f1782eb6be1dba0bdf4621a3d1cc6a7611ec58702c8.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/tables/421bc948a42b4e6643c80f1782eb6be1dba0bdf4621a3d1cc6a7611ec58702c8.jpg)

![c90f7159228accb9ee6b91b949d88cdfb45617d3ffce72ac6e478cc6ddab10f7.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/tables/c90f7159228accb9ee6b91b949d88cdfb45617d3ffce72ac6e478cc6ddab10f7.jpg)

![d6ab2f7b0c7c3325358eca0023cc4b6e1aec716e399928326cae9cea1241572f.jpg](../icml_results/705_D-ARL_%20A%20Distribution-Matched%20Asynchronous%20Reinforcement%20Learning%20Framework%20for%20Language%20Reasoning/tables/d6ab2f7b0c7c3325358eca0023cc4b6e1aec716e399928326cae9cea1241572f.jpg)

<a id="paper-0696"></a>

## Diving into Kronecker Adapters: Component Design Matters


### Images

![555f349d8241adf2beb970e7269b81cd66d0cfcbe3ecf68203ce26cd3b7a7c28.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/images/555f349d8241adf2beb970e7269b81cd66d0cfcbe3ecf68203ce26cd3b7a7c28.jpg)

### Tables

![045cce7a3dfdcb3e837a0a7e78c32028d952e5454b456ac9668d775b2c748aec.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/045cce7a3dfdcb3e837a0a7e78c32028d952e5454b456ac9668d775b2c748aec.jpg)

![1d15a3ee0bdcf13b76ca4ab72689dccbbb874a3ff3d3b21de6b1a4984187134d.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/1d15a3ee0bdcf13b76ca4ab72689dccbbb874a3ff3d3b21de6b1a4984187134d.jpg)

![2e81fe7708f66cca2ecc18b68630d1f29fee1a0780d1a350b5181815011c0ad2.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/2e81fe7708f66cca2ecc18b68630d1f29fee1a0780d1a350b5181815011c0ad2.jpg)

![336e13485afdd6727cd9468fa41e376dfc3c4b814d14576eff50cbb30a5fe73d.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/336e13485afdd6727cd9468fa41e376dfc3c4b814d14576eff50cbb30a5fe73d.jpg)

![40f3bfff929ec8d64be4916973131e6918893c7d30c71976246df0372b36f038.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/40f3bfff929ec8d64be4916973131e6918893c7d30c71976246df0372b36f038.jpg)

![4332808bffb0ff872223d5e2c78288c17cac018a76ac4d7247a274cf3b92b7f8.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/4332808bffb0ff872223d5e2c78288c17cac018a76ac4d7247a274cf3b92b7f8.jpg)

![43523798aa2fee2f3ae2bf6858a9ca0d8f0811957073e1dd4bdcb1d90b844b3e.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/43523798aa2fee2f3ae2bf6858a9ca0d8f0811957073e1dd4bdcb1d90b844b3e.jpg)

![509d8b5b5b9b1c3f2945c8a71e7f78c35e0815ec9b968cdc0e3844093babb2e7.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/509d8b5b5b9b1c3f2945c8a71e7f78c35e0815ec9b968cdc0e3844093babb2e7.jpg)

![5bcac525ed7675c1d4e9dc2a65c48c6a3d6a26fec449aa78c34a8e228aa4446a.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/5bcac525ed7675c1d4e9dc2a65c48c6a3d6a26fec449aa78c34a8e228aa4446a.jpg)

![5d2271d48fc6309793e71bd5b1898d45061f0cc2b060f783bad17d7aa5e91a63.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/5d2271d48fc6309793e71bd5b1898d45061f0cc2b060f783bad17d7aa5e91a63.jpg)

![62bbd2a88a8e083dab8495ac262ccec0293670c0f329829ba043725859dbdd43.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/62bbd2a88a8e083dab8495ac262ccec0293670c0f329829ba043725859dbdd43.jpg)

![7cd044c7d9a8e265415458a9bde33f344ef799b5dff2705110b702987eb2f45d.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/7cd044c7d9a8e265415458a9bde33f344ef799b5dff2705110b702987eb2f45d.jpg)

![7f030cd80e111faeecd9d5d2a1596541a4a0d49aca675c3a5dc9224151150495.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/7f030cd80e111faeecd9d5d2a1596541a4a0d49aca675c3a5dc9224151150495.jpg)

![832a9dd39ab1efe96d36ec11c56096ab288902877f1548d0451783cd491f32b4.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/832a9dd39ab1efe96d36ec11c56096ab288902877f1548d0451783cd491f32b4.jpg)

![9531ac31fc8fffc75094f700c3907e12387fa0710ec03ed33555c7d914f84a77.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/9531ac31fc8fffc75094f700c3907e12387fa0710ec03ed33555c7d914f84a77.jpg)

![98386c504303d04a9add0cc7b91f09664c926bc0a541ee32f68f8fb31c9e9b79.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/98386c504303d04a9add0cc7b91f09664c926bc0a541ee32f68f8fb31c9e9b79.jpg)

![9a17fb35d2a804d868a3770a2ac285c323683b1262cc8408661f7a86ebcd93db.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/9a17fb35d2a804d868a3770a2ac285c323683b1262cc8408661f7a86ebcd93db.jpg)

![be4f7b312f96b035e325b98666ab73a44eb3e17e80c077b210c98491b6da5703.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/be4f7b312f96b035e325b98666ab73a44eb3e17e80c077b210c98491b6da5703.jpg)

![c28844e151ba32350f2e965d0e9ac823efd9e9aa134cfc7fade99883585244b0.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/c28844e151ba32350f2e965d0e9ac823efd9e9aa134cfc7fade99883585244b0.jpg)

![ca0a16d68c97bad3af5bb605cd17021eaee0dd98ee15792c3e80122cab26b3bc.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/ca0a16d68c97bad3af5bb605cd17021eaee0dd98ee15792c3e80122cab26b3bc.jpg)

![ce68ed0d03ac66d8006d7610a41d7f8640abeaa3cc0073e16d6ea8172b2309a6.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/ce68ed0d03ac66d8006d7610a41d7f8640abeaa3cc0073e16d6ea8172b2309a6.jpg)

![d72625fc13935bf624cfc7efb03acf67b42b3fdc68224ff03d041e1499dcd96f.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/d72625fc13935bf624cfc7efb03acf67b42b3fdc68224ff03d041e1499dcd96f.jpg)

![daa2562763375745259c1d7a00de4495c80b3e999b3c2d476d02a6b802299392.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/daa2562763375745259c1d7a00de4495c80b3e999b3c2d476d02a6b802299392.jpg)

![db6d5d7c9d876031ddd6f1ee17d939bc1c7a43b7c79865c39f6b7a7329e8635d.jpg](../icml_results/706_Diving%20into%20Kronecker%20Adapters_%20Component%20Design%20Matters/tables/db6d5d7c9d876031ddd6f1ee17d939bc1c7a43b7c79865c39f6b7a7329e8635d.jpg)

<a id="paper-0697"></a>

## Optimizing Few-Step Generation with Adaptive Matching Distillation


### Images

![1c7fc7edc9f62f378488359891f05456ced31d369bad9a5267186a7d37b022c1.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/1c7fc7edc9f62f378488359891f05456ced31d369bad9a5267186a7d37b022c1.jpg)

![2d8d050eb9377c457511297d710b2124f48ce1d9d4e729985d99e7e711f2aac7.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/2d8d050eb9377c457511297d710b2124f48ce1d9d4e729985d99e7e711f2aac7.jpg)

![3676cd57ff44e1c7c9ec1d2105db76a2062f3d3626bb163f45b8b215b5c4861f.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/3676cd57ff44e1c7c9ec1d2105db76a2062f3d3626bb163f45b8b215b5c4861f.jpg)

![648298b57c4dfb6543602f95123fc861d5d594248cfa65c2147f059d54a1b9f6.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/648298b57c4dfb6543602f95123fc861d5d594248cfa65c2147f059d54a1b9f6.jpg)

![6d8035f8d43a85f79120bf4e9c78062e10d3b80ddddd2853f1e640d341e7be86.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/6d8035f8d43a85f79120bf4e9c78062e10d3b80ddddd2853f1e640d341e7be86.jpg)

![7463e404d479d54d520416c8c14334fd246a8f55419960c66f5c2de598461871.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/7463e404d479d54d520416c8c14334fd246a8f55419960c66f5c2de598461871.jpg)

![7641dad047ac15921f777de30174499428081d5b58e6300723f87a3381973193.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/7641dad047ac15921f777de30174499428081d5b58e6300723f87a3381973193.jpg)

![a01987454d9f71ea5223942bf87c0547197a83617faa995d25ca97a2ab3626f8.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/a01987454d9f71ea5223942bf87c0547197a83617faa995d25ca97a2ab3626f8.jpg)

![b307c7b01a5641f807a565e4095efd7fe3039131a6dfd08c2b7561fb94bab6e4.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/b307c7b01a5641f807a565e4095efd7fe3039131a6dfd08c2b7561fb94bab6e4.jpg)

![d0c243f621112836503437043275d069a64f53cc345ae6991eddd60c488dc806.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/d0c243f621112836503437043275d069a64f53cc345ae6991eddd60c488dc806.jpg)

![dc14ef658a31b390833ac2117ad840f286d333271ca6f84b42838b645e7053f8.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/dc14ef658a31b390833ac2117ad840f286d333271ca6f84b42838b645e7053f8.jpg)

![df044c3e600cd91bc3f00e691922e50b9118ee8db56e865456c62c63fa2c03da.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/df044c3e600cd91bc3f00e691922e50b9118ee8db56e865456c62c63fa2c03da.jpg)

![e8f9994e2ba1ea1c2150246ef8fb18ee3bd9a17145b23cc0c0d2713c7ce15f7a.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/e8f9994e2ba1ea1c2150246ef8fb18ee3bd9a17145b23cc0c0d2713c7ce15f7a.jpg)

![ee38b6a33864421b86596d5e82550912482fedb2d2b18ff160ecdc8ebff6d2ae.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/ee38b6a33864421b86596d5e82550912482fedb2d2b18ff160ecdc8ebff6d2ae.jpg)

![efad74fba4dd083838515d8c20901e7ad9d9155ec900ff5cd175cbc5db55cbd8.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/images/efad74fba4dd083838515d8c20901e7ad9d9155ec900ff5cd175cbc5db55cbd8.jpg)

### Tables

![09fcccdf9d42189272ec0ae732ae7f6ddd92fa3499d07608d3362c7262b9dd50.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/09fcccdf9d42189272ec0ae732ae7f6ddd92fa3499d07608d3362c7262b9dd50.jpg)

![2db90751d49f5e1f0d2d65de7e0c76447678082100a46640eb169473137a98c2.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/2db90751d49f5e1f0d2d65de7e0c76447678082100a46640eb169473137a98c2.jpg)

![3cf8b35e8dd3c2b037f32e881240575fc18c5e3270ac12055ee007ea54780cad.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/3cf8b35e8dd3c2b037f32e881240575fc18c5e3270ac12055ee007ea54780cad.jpg)

![44d6ab2d14f28fa12487dc911232844a4431ff8900c276aadda8ff11bbb6218b.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/44d6ab2d14f28fa12487dc911232844a4431ff8900c276aadda8ff11bbb6218b.jpg)

![7d43ff353ffa2a45ea7526a65b462507b8cf53e5e794cda400729a9df1b216ae.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/7d43ff353ffa2a45ea7526a65b462507b8cf53e5e794cda400729a9df1b216ae.jpg)

![82001577c5a6206dc7765d4372a28019ec79d43b44762d2bb8f69377ef1ee82c.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/82001577c5a6206dc7765d4372a28019ec79d43b44762d2bb8f69377ef1ee82c.jpg)

![b2ef0bb48ef07a78517dbecf1a32980ac25a66ff3e1a9a61dd8f1e000519b971.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/b2ef0bb48ef07a78517dbecf1a32980ac25a66ff3e1a9a61dd8f1e000519b971.jpg)

![c49f8e75a69d9a0254f4ce29391097ed69a68a7b4391cabb8f6f1aa8a2e6d98a.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/c49f8e75a69d9a0254f4ce29391097ed69a68a7b4391cabb8f6f1aa8a2e6d98a.jpg)

![c9ee07176356cf77d127a8c4c249a70fca6d1621f6ec455408a87b037f84f7d8.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/c9ee07176356cf77d127a8c4c249a70fca6d1621f6ec455408a87b037f84f7d8.jpg)

![e5fda67e2391f056af45bcd4d011c6754c5fed3a8f67568bfe3dea0dcb97c181.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/e5fda67e2391f056af45bcd4d011c6754c5fed3a8f67568bfe3dea0dcb97c181.jpg)

![f21b9ea4d6c87d200e36bec199ba941653d16683577f1a34860b2152713a3c6e.jpg](../icml_results/707_Optimizing%20Few-Step%20Generation%20with%20Adaptive%20Matching%20Distillation/tables/f21b9ea4d6c87d200e36bec199ba941653d16683577f1a34860b2152713a3c6e.jpg)

<a id="paper-0698"></a>

## Adaptive Preconditioners Trigger Loss Spikes in Adam


### Images

![0089f4ea9883ca3aa57ad0c395f5cd939d497b54d9d133ade10ade0869671902.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/0089f4ea9883ca3aa57ad0c395f5cd939d497b54d9d133ade10ade0869671902.jpg)

![03aff8ccd2f6ba2d8b651dbdb6bb62d233039e7190fbeb8820686d2a12309c5a.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/03aff8ccd2f6ba2d8b651dbdb6bb62d233039e7190fbeb8820686d2a12309c5a.jpg)

![07a178722461794f12a6caf25166c08ea01340c8328194c74b1a817dc73d9e94.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/07a178722461794f12a6caf25166c08ea01340c8328194c74b1a817dc73d9e94.jpg)

![128fa4c2d4df981de9b3095dcba51416d965c88810307ba81f502011db9b336c.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/128fa4c2d4df981de9b3095dcba51416d965c88810307ba81f502011db9b336c.jpg)

![15fddbfc599248cd840d1f31730cb5a1ac12104fad560fb19d8574e698d173a0.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/15fddbfc599248cd840d1f31730cb5a1ac12104fad560fb19d8574e698d173a0.jpg)

![304054069b0fe0f7e3bc708576e2a316317cfc33d7a39e165ffea0b9ba5f67bd.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/304054069b0fe0f7e3bc708576e2a316317cfc33d7a39e165ffea0b9ba5f67bd.jpg)

![3e014cbe3bf5c58c3f0c0f438abc5b85a40c606d6aaa478866e48738d0e46639.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/3e014cbe3bf5c58c3f0c0f438abc5b85a40c606d6aaa478866e48738d0e46639.jpg)

![45f7e62eeba8b072fa24ce30528d96f9f25c468f3396f85f59f6a54a6273fb94.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/45f7e62eeba8b072fa24ce30528d96f9f25c468f3396f85f59f6a54a6273fb94.jpg)

![4dcdb4f9d6e9fed25633101b1775fd5e1d97d93a497c776fec54999b4b9032ca.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/4dcdb4f9d6e9fed25633101b1775fd5e1d97d93a497c776fec54999b4b9032ca.jpg)

![5164de74626672b73d39da535b999d765ef97974f1c97d9a45dd62003e5c2550.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/5164de74626672b73d39da535b999d765ef97974f1c97d9a45dd62003e5c2550.jpg)

![5e411560fc69e116a12779d3e096ce9e7eb97513a2a7235c93f943cd7e61a5c5.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/5e411560fc69e116a12779d3e096ce9e7eb97513a2a7235c93f943cd7e61a5c5.jpg)

![5e635799242ca33085e8f5077cd6d1430fbd952167be5eca37a057932c9db81d.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/5e635799242ca33085e8f5077cd6d1430fbd952167be5eca37a057932c9db81d.jpg)

![62eb36224f64ce4c5d87e46ed813d15ab4cefff9f4613c00f7f666f46c9faeef.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/62eb36224f64ce4c5d87e46ed813d15ab4cefff9f4613c00f7f666f46c9faeef.jpg)

![715af516836d5e0194f1035792f55346e370e908224e809f6dfb5e04d584f1e6.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/715af516836d5e0194f1035792f55346e370e908224e809f6dfb5e04d584f1e6.jpg)

![78ac02c2390049649a39ba76c8332868b13941a318cf085a8501b2d7719cc28e.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/78ac02c2390049649a39ba76c8332868b13941a318cf085a8501b2d7719cc28e.jpg)

![8369982a17d16b4ef65a1ef7e42bb0eb2fcc8b326af30e2affe5961b3c4f0615.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/8369982a17d16b4ef65a1ef7e42bb0eb2fcc8b326af30e2affe5961b3c4f0615.jpg)

![8a05a71a51187562c2f8f43c170f33f0b28ee46840c3e245e9ef5f0a31456ce2.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/8a05a71a51187562c2f8f43c170f33f0b28ee46840c3e245e9ef5f0a31456ce2.jpg)

![95fa91aad40998526551aa5668734573552eb1665209bd808340e21257591bb8.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/95fa91aad40998526551aa5668734573552eb1665209bd808340e21257591bb8.jpg)

![9da1349f23061b95bf5d94381d0ccb4939cff52e6f307ee13bd4f8004e123aac.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/9da1349f23061b95bf5d94381d0ccb4939cff52e6f307ee13bd4f8004e123aac.jpg)

![a299ec2fad3fffd99f25675fc100b409715c85059f690807bea27ba72ca94710.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/a299ec2fad3fffd99f25675fc100b409715c85059f690807bea27ba72ca94710.jpg)

![af8b7328c79916f2732539449b9af62be51af51914dbda5acd6e6820fe9fd313.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/af8b7328c79916f2732539449b9af62be51af51914dbda5acd6e6820fe9fd313.jpg)

![b3892c8c38e94ffd1a9e3ade157ffaa69344eb1786e3eacb144b40119736f6a9.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/b3892c8c38e94ffd1a9e3ade157ffaa69344eb1786e3eacb144b40119736f6a9.jpg)

![b82c61a893b90a256d36c1fa2fd1921a9322785c17c61637eb286827133a01e4.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/b82c61a893b90a256d36c1fa2fd1921a9322785c17c61637eb286827133a01e4.jpg)

![c0e37e28b7fd474f3347674f49c0518564fa24d998eaa7813905675aaf724156.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/c0e37e28b7fd474f3347674f49c0518564fa24d998eaa7813905675aaf724156.jpg)

![c76ac017f2bb88f495b2b8e35c758cc938d9e0538c86b4f977fbdad7ccfc3130.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/c76ac017f2bb88f495b2b8e35c758cc938d9e0538c86b4f977fbdad7ccfc3130.jpg)

![cbfa56f40e334e0f833e28fbaeed1cf0dca58c0a4b64b8b79eb2b0554f3f1581.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/cbfa56f40e334e0f833e28fbaeed1cf0dca58c0a4b64b8b79eb2b0554f3f1581.jpg)

![f389ac0feeaddc533529d3e3a781ed67eaa18be16b72e8dd2521de509a76763e.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/f389ac0feeaddc533529d3e3a781ed67eaa18be16b72e8dd2521de509a76763e.jpg)

![f3e3af81c156c0181bc5e31aa00a105b4cd5e17c1f3804fd5cf7d7d1e71f68a7.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/f3e3af81c156c0181bc5e31aa00a105b4cd5e17c1f3804fd5cf7d7d1e71f68a7.jpg)

![f6ca17b540d019e86091025649f63b31803dc26b998179dc252e7401117279cc.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/f6ca17b540d019e86091025649f63b31803dc26b998179dc252e7401117279cc.jpg)

![fead64ea3918d70a1c7ecd5e46a178079ad0e20f86bedb2e66d842eb421ce65e.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/fead64ea3918d70a1c7ecd5e46a178079ad0e20f86bedb2e66d842eb421ce65e.jpg)

![fef6e3d85aa2bb6af75352851ae1d415728dbd6cecb9439051fe368adc5cfb5b.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/images/fef6e3d85aa2bb6af75352851ae1d415728dbd6cecb9439051fe368adc5cfb5b.jpg)

### Tables

![ec3ce92517d205a42eadb6e83e5f7ab5e641849bee02b325cc2b9b5bd420895e.jpg](../icml_results/708_Adaptive%20Preconditioners%20Trigger%20Loss%20Spikes%20in%20Adam/tables/ec3ce92517d205a42eadb6e83e5f7ab5e641849bee02b325cc2b9b5bd420895e.jpg)

<a id="paper-0699"></a>

## Compressed Sensing for Capability Localization in Large Language Models


### Images

![16278eff12094ab9a1a58fdc0575bfed46e45b7c17d3e5743e670d05342af175.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/images/16278eff12094ab9a1a58fdc0575bfed46e45b7c17d3e5743e670d05342af175.jpg)

![1a4d29ee6c2c9b5effa98824dbc0fb33d1f0c50d95deb2423fc52c111a930e9d.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/images/1a4d29ee6c2c9b5effa98824dbc0fb33d1f0c50d95deb2423fc52c111a930e9d.jpg)

![679273371960829de050f00028a78526ebaf8272276a93a981adfed60b4717b0.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/images/679273371960829de050f00028a78526ebaf8272276a93a981adfed60b4717b0.jpg)

![7928206103bd85208dcd8ebec4d964886ca98bfece4310035ad330080b43e619.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/images/7928206103bd85208dcd8ebec4d964886ca98bfece4310035ad330080b43e619.jpg)

### Tables

![1dbc94ddb413da07a7a8c069f817383eb31f613ed239dfe72b1209d51d614b7f.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/1dbc94ddb413da07a7a8c069f817383eb31f613ed239dfe72b1209d51d614b7f.jpg)

![58b96268ed3ee4e77a4f1276a7084a62faf611263958a7a1439ec03ab278882f.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/58b96268ed3ee4e77a4f1276a7084a62faf611263958a7a1439ec03ab278882f.jpg)

![62cc43771caca4c1fa983c62cf12d524404d79a97e2c9324eb7695d5920d3288.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/62cc43771caca4c1fa983c62cf12d524404d79a97e2c9324eb7695d5920d3288.jpg)

![672217e974f12978b9f134a3e9df91d9e2f0aae53653f21e885b2774f24430ce.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/672217e974f12978b9f134a3e9df91d9e2f0aae53653f21e885b2774f24430ce.jpg)

![6a9de22a5771ce8cebe7e0c5c0d1afc98d60943cf40baa84c534c879e0d8543b.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/6a9de22a5771ce8cebe7e0c5c0d1afc98d60943cf40baa84c534c879e0d8543b.jpg)

![84342d25d670eeb36721387c5697e3d02b3e21415a58efcd840b25b97cdba9d3.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/84342d25d670eeb36721387c5697e3d02b3e21415a58efcd840b25b97cdba9d3.jpg)

![9106f62a7ef867e092ea86a3dca8a77596baebed10508bbbb1893eeabb27d9cc.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/9106f62a7ef867e092ea86a3dca8a77596baebed10508bbbb1893eeabb27d9cc.jpg)

![9ba66294d5156c000c80d286815c0a51b69e96154d2487a70df6f7b3169505c3.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/9ba66294d5156c000c80d286815c0a51b69e96154d2487a70df6f7b3169505c3.jpg)

![9bbfc146225e0eab970cc65182430d6620ad6fe643c83a154798373c53a2b2d9.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/9bbfc146225e0eab970cc65182430d6620ad6fe643c83a154798373c53a2b2d9.jpg)

![9e5d922273028249662a151d05a19d14df8954ca508ab6786ca20e6952de0b71.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/9e5d922273028249662a151d05a19d14df8954ca508ab6786ca20e6952de0b71.jpg)

![a35544b823c31a4b2175531c7cab4fde1ec1f4ad7ef2e90f678972481784fe86.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/a35544b823c31a4b2175531c7cab4fde1ec1f4ad7ef2e90f678972481784fe86.jpg)

![ccf2a45441a87e2b31969e46cac2391db3f4a4b1d1b1805befafe28b9be619cd.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/ccf2a45441a87e2b31969e46cac2391db3f4a4b1d1b1805befafe28b9be619cd.jpg)

![cd83fef6e8a0a3fc422f7b008197d3aa24d01891efe6b85d29ae6cec650b6af6.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/cd83fef6e8a0a3fc422f7b008197d3aa24d01891efe6b85d29ae6cec650b6af6.jpg)

![e223920ce00d566dbfd4b93b36b6dc68d44fb7e51791c1f3813aaf6b7d4e3513.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/e223920ce00d566dbfd4b93b36b6dc68d44fb7e51791c1f3813aaf6b7d4e3513.jpg)

![e8d5fde72c0711a98a1f9dfa7d5fb54cbab57b8a54b89833548e79ba97a7d25e.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/e8d5fde72c0711a98a1f9dfa7d5fb54cbab57b8a54b89833548e79ba97a7d25e.jpg)

![ea4b1ad782e575b109f66590ffc339d25919050bb32f863dcadfd0dc0bed70f8.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/ea4b1ad782e575b109f66590ffc339d25919050bb32f863dcadfd0dc0bed70f8.jpg)

![f6e5263ba9f3beeec48a45a38247b4b1aa504669e378a93dc69d38f4ad55fcad.jpg](../icml_results/709_Compressed%20Sensing%20for%20Capability%20Localization%20in%20Large%20Language%20Models/tables/f6e5263ba9f3beeec48a45a38247b4b1aa504669e378a93dc69d38f4ad55fcad.jpg)

<a id="paper-0700"></a>

## Supervised Guidance Training for Infinite-Dimensional Diffusion Models


### Images

![22b0906010b8b66038b58fcaf42bc9cbfa5b8fbcd245cee9c7155f2cd122f20d.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/22b0906010b8b66038b58fcaf42bc9cbfa5b8fbcd245cee9c7155f2cd122f20d.jpg)

![6dc7ff886a08bb40a63e09c6d39a7de9880708773588d1b862cdf31c5fa46935.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/6dc7ff886a08bb40a63e09c6d39a7de9880708773588d1b862cdf31c5fa46935.jpg)

![a4269400e71c618007578fe51195d6cece8a4574422f4dd2a0660b7b2aa779f6.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/a4269400e71c618007578fe51195d6cece8a4574422f4dd2a0660b7b2aa779f6.jpg)

![ba45c31e6dbf00e7d0ed905467646b7e56b1f1fa72d668bd5626d1ebe14603f5.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/ba45c31e6dbf00e7d0ed905467646b7e56b1f1fa72d668bd5626d1ebe14603f5.jpg)

![c5e5a34e9f0e49c46eb52fdf86b12b8c6cbe7a50e99dc1d28b1cb4daade146b0.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/c5e5a34e9f0e49c46eb52fdf86b12b8c6cbe7a50e99dc1d28b1cb4daade146b0.jpg)

![cc553c326c5d1fd0056d75a69d56057d5b38205a506df4c9af45ade219d0b2a2.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/cc553c326c5d1fd0056d75a69d56057d5b38205a506df4c9af45ade219d0b2a2.jpg)

![ce866a28aeb4cbd62e27b6fffcee510f965488fc9c3e2878ba911f3c482d4e54.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/ce866a28aeb4cbd62e27b6fffcee510f965488fc9c3e2878ba911f3c482d4e54.jpg)

![e76bc46f6c49af5cc41def10a85d43345000c656d620bdbda1889c3576dc9151.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/images/e76bc46f6c49af5cc41def10a85d43345000c656d620bdbda1889c3576dc9151.jpg)

### Tables

![0239d6e402570f538f31b4a4bcde4127176cb9106cca1c18870912cd187ea8bb.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/tables/0239d6e402570f538f31b4a4bcde4127176cb9106cca1c18870912cd187ea8bb.jpg)

![25e900e01e0d4488bb1548b88149d5541cea7989205e71fb7b002caeb62d6764.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/tables/25e900e01e0d4488bb1548b88149d5541cea7989205e71fb7b002caeb62d6764.jpg)

![38082669e7aab92ec40c60f59474cf7bd802d9e825d485aeea4b3ebc8577ff35.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/tables/38082669e7aab92ec40c60f59474cf7bd802d9e825d485aeea4b3ebc8577ff35.jpg)

![3d4f9489f5e711820a0a31c7877c8d37e09c06e7cdd3e0d0a531f3e2a72507ca.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/tables/3d4f9489f5e711820a0a31c7877c8d37e09c06e7cdd3e0d0a531f3e2a72507ca.jpg)

![82a6f69b262b6d2fdc0dda1a8d42788412d3b8b79db8564138155cb5dad77013.jpg](../icml_results/710_Supervised%20Guidance%20Training%20for%20Infinite-Dimensional%20Diffusion%20Models/tables/82a6f69b262b6d2fdc0dda1a8d42788412d3b8b79db8564138155cb5dad77013.jpg)

<a id="paper-0701"></a>

## Hair-Trigger Alignment: Black-Box Evaluation Cannot Guarantee Post-Update Alignment


### Images

![a3a2dfe61f4f554fb80b7381518ea44c1a7a36dd7fcc3eaaf63c23c5c3a2a44d.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/images/a3a2dfe61f4f554fb80b7381518ea44c1a7a36dd7fcc3eaaf63c23c5c3a2a44d.jpg)

![b77ff759174ea30c8c7350e3be0d3dc39295214eab72cac268e955bb3d7eec16.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/images/b77ff759174ea30c8c7350e3be0d3dc39295214eab72cac268e955bb3d7eec16.jpg)

![caca10514e8785edcfe55cd23d3483baa9eb3aabe5187bf7d5ad473270aaf66a.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/images/caca10514e8785edcfe55cd23d3483baa9eb3aabe5187bf7d5ad473270aaf66a.jpg)

![d00dfea2ab35aaff70ca7e22c0142d82f3e863f951dad7628ae40e08d0495551.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/images/d00dfea2ab35aaff70ca7e22c0142d82f3e863f951dad7628ae40e08d0495551.jpg)

### Tables

![0730d89f6124eea0318935473cc569f207bd6062eff690ab26e57bbad9ecc7c5.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/0730d89f6124eea0318935473cc569f207bd6062eff690ab26e57bbad9ecc7c5.jpg)

![168ab8a1d257c2fad615a92b03e02ecfa9d9a378785d52b466e801906d3baed1.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/168ab8a1d257c2fad615a92b03e02ecfa9d9a378785d52b466e801906d3baed1.jpg)

![7c8d105bf50cfb57a24d2eda0405ef487c6526af92786a8805971fdac7eb7a99.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/7c8d105bf50cfb57a24d2eda0405ef487c6526af92786a8805971fdac7eb7a99.jpg)

![8cf4ab13c02b99f0aee4692a82028f262731073967b9f3d721b57aa83a59ae31.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/8cf4ab13c02b99f0aee4692a82028f262731073967b9f3d721b57aa83a59ae31.jpg)

![9bf0e74d1922ef5716e3423d1220a50d3bf9e362f4862a1d9a9276a9a5f3e929.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/9bf0e74d1922ef5716e3423d1220a50d3bf9e362f4862a1d9a9276a9a5f3e929.jpg)

![b89c2d63a18fafd7cc83e1c6a8b10a9d67382a107960320782e56062c017b848.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/b89c2d63a18fafd7cc83e1c6a8b10a9d67382a107960320782e56062c017b848.jpg)

![bdcf5a3f58dc2a8cfac923504431fb8be6b87d051ba01ba2c5a4b3c581b7b38d.jpg](../icml_results/711_Hair-Trigger%20Alignment_%20Black-Box%20Evaluation%20Cannot%20Guarantee%20Post-Update%20Alignment/tables/bdcf5a3f58dc2a8cfac923504431fb8be6b87d051ba01ba2c5a4b3c581b7b38d.jpg)

<a id="paper-0702"></a>

## Collaborative Threshold Watermarking


### Images

![102ca661a75c7099e14c39d262d873b698d93619d0c812279ab266bab793646e.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/102ca661a75c7099e14c39d262d873b698d93619d0c812279ab266bab793646e.jpg)

![110b0f1dbade69e3166cc61a98127b791e79ed26abb134e08037057c775b36a6.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/110b0f1dbade69e3166cc61a98127b791e79ed26abb134e08037057c775b36a6.jpg)

![25f1901ae99f235172fd8015d268f1dec9fff3ae66f1190063fe560defce1b0c.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/25f1901ae99f235172fd8015d268f1dec9fff3ae66f1190063fe560defce1b0c.jpg)

![337a51c5dfa8cb31522ceb3e0eb7726f7a824945650f5932bf9042b42de0659c.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/337a51c5dfa8cb31522ceb3e0eb7726f7a824945650f5932bf9042b42de0659c.jpg)

![35fa4c57953019efcbc0c7e2b6dd9821832bf0a7b0aedcf28a7c92378a138a10.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/35fa4c57953019efcbc0c7e2b6dd9821832bf0a7b0aedcf28a7c92378a138a10.jpg)

![4071dceb0a1300f32a8c1f9bf2748d9c641034ef3ef28916e7d664fc13325780.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/4071dceb0a1300f32a8c1f9bf2748d9c641034ef3ef28916e7d664fc13325780.jpg)

![47af02b914ae011397c8c94e26555660848882d51d71d040b3ee234018ba3a48.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/47af02b914ae011397c8c94e26555660848882d51d71d040b3ee234018ba3a48.jpg)

![8539fcaaa3fb8b65069e98dd881e495af65c3b065664b87f6643de7246ae8a07.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/8539fcaaa3fb8b65069e98dd881e495af65c3b065664b87f6643de7246ae8a07.jpg)

![988c1301a658b0b557aff274915443e05da76afa6183debc65e1d969ae3a9f47.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/988c1301a658b0b557aff274915443e05da76afa6183debc65e1d969ae3a9f47.jpg)

![99d19d25fa2e8eb69c3e09da322e806d5963decae15b609d242e6889a1d5c0be.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/99d19d25fa2e8eb69c3e09da322e806d5963decae15b609d242e6889a1d5c0be.jpg)

![acd3af39d724bb15de52c7cb67fbeb6096f9685afdceb8f9c8d5873706b174ad.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/acd3af39d724bb15de52c7cb67fbeb6096f9685afdceb8f9c8d5873706b174ad.jpg)

![be6f242626864060089cf99550dce5c5213145e84265ff9a28d2f542174356e0.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/images/be6f242626864060089cf99550dce5c5213145e84265ff9a28d2f542174356e0.jpg)

### Tables

![2d3235041070995d09e2e5ab64f070ebd0224896149d13c055df05178a67c03d.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/tables/2d3235041070995d09e2e5ab64f070ebd0224896149d13c055df05178a67c03d.jpg)

![397442bedc92cc188f97013d5df15828dc7a0b659d7707f7cfd949147a8d01b8.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/tables/397442bedc92cc188f97013d5df15828dc7a0b659d7707f7cfd949147a8d01b8.jpg)

![cba983125d46b9632101945227402e6b790890946ebf9383bc490ff78dfd2bd0.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/tables/cba983125d46b9632101945227402e6b790890946ebf9383bc490ff78dfd2bd0.jpg)

![f0f75e3df8acdd81074b5478d6ce88fd79bf0d60c0bfff8e291fd3e2d9af83ed.jpg](../icml_results/712_Collaborative%20Threshold%20Watermarking/tables/f0f75e3df8acdd81074b5478d6ce88fd79bf0d60c0bfff8e291fd3e2d9af83ed.jpg)

<a id="paper-0703"></a>

## miniF2F-Dafny: LLM-Guided Mathematical Theorem Proving via Auto-Active Verification


### Images

![09baa471c08b3571dc4bebc7dcf4458db080aff36a60b80c159aa0a3a6f0c813.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/09baa471c08b3571dc4bebc7dcf4458db080aff36a60b80c159aa0a3a6f0c813.jpg)

![3d99f074f8c7e87f8fd0e228cf860253d4cbf1650a1140588c2ee07a069611b3.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/3d99f074f8c7e87f8fd0e228cf860253d4cbf1650a1140588c2ee07a069611b3.jpg)

![4b076763ffa30752b7f72cb9ca882061cc7cab513e175c43f3b5a6053aea0cce.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/4b076763ffa30752b7f72cb9ca882061cc7cab513e175c43f3b5a6053aea0cce.jpg)

![8cd7888215a8df150a4bca1de436caec65e8c66a02e41c1e45df2df394fd05c5.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/8cd7888215a8df150a4bca1de436caec65e8c66a02e41c1e45df2df394fd05c5.jpg)

![9b7128b6c0f1bd9b2507b4bc9b633db033a3cc9ea9e4cd2d607fea71868f012d.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/9b7128b6c0f1bd9b2507b4bc9b633db033a3cc9ea9e4cd2d607fea71868f012d.jpg)

![c80ec8c5b157fd6457b3303ffa25b77e356f98c3d551389fb2bcb0a16afc7664.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/c80ec8c5b157fd6457b3303ffa25b77e356f98c3d551389fb2bcb0a16afc7664.jpg)

![ce8773eb3df9e8a953b4e2f63514c0c4232f578df35d53514b0ec3e55918bd4a.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/ce8773eb3df9e8a953b4e2f63514c0c4232f578df35d53514b0ec3e55918bd4a.jpg)

![ecc9c7633ea5cb779528c2f57f3c267872af8e18224e19523d53ef0b07dcc941.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/ecc9c7633ea5cb779528c2f57f3c267872af8e18224e19523d53ef0b07dcc941.jpg)

![fbe3b65d7a7b4acb65bf80bb393bd783bbade3f42f3eafb3c2ef9cb4caa52089.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/images/fbe3b65d7a7b4acb65bf80bb393bd783bbade3f42f3eafb3c2ef9cb4caa52089.jpg)

### Tables

![625bf89f6b9e4d1cb1f0bc0e35a849ae7b098f809fb29bed3504f896eac21d1e.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/tables/625bf89f6b9e4d1cb1f0bc0e35a849ae7b098f809fb29bed3504f896eac21d1e.jpg)

![8390f21bf43780c31ce4cbc6bbcc70ff685db4edd964be738a37f276567d5fde.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/tables/8390f21bf43780c31ce4cbc6bbcc70ff685db4edd964be738a37f276567d5fde.jpg)

![bab86fce1972d3d048afd3116d8093f7b2da5509a146ef99a773ffa14e630290.jpg](../icml_results/713_miniF2F-Dafny_%20LLM-Guided%20Mathematical%20Theorem%20Proving%20via%20Auto-Active%20Verification/tables/bab86fce1972d3d048afd3116d8093f7b2da5509a146ef99a773ffa14e630290.jpg)

<a id="paper-0704"></a>

## Fixed Budget is No Harder Than Fixed Confidence in Best-Arm Identification up to Logarithmic Factors


### Images

![18da6617ed1d20953037a7314f11b07e052b5594105f5edc0fb9b0d828c3c0b5.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/images/18da6617ed1d20953037a7314f11b07e052b5594105f5edc0fb9b0d828c3c0b5.jpg)

![2cafb9df7058d5530b0025872196618f0997b0abc0fdadf0d21e00bdcdaaf55e.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/images/2cafb9df7058d5530b0025872196618f0997b0abc0fdadf0d21e00bdcdaaf55e.jpg)

![4499a6bc075348b1a1cca72e7ea66bd17e4fe4e5861313edd83b40fbed22f3e3.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/images/4499a6bc075348b1a1cca72e7ea66bd17e4fe4e5861313edd83b40fbed22f3e3.jpg)

![6141be7b5396089b850e7729ee4464135cd706835b8f7b9fc79e6c2f0f88897d.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/images/6141be7b5396089b850e7729ee4464135cd706835b8f7b9fc79e6c2f0f88897d.jpg)

![eee74b0709f147668c70c1069c77b1a9f6c74d61529730f33205044421dac911.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/images/eee74b0709f147668c70c1069c77b1a9f6c74d61529730f33205044421dac911.jpg)

### Tables

![74ca8597fbfc8ce874330899d65f7f6e6b9144685fde01fd3380609884d181bf.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/tables/74ca8597fbfc8ce874330899d65f7f6e6b9144685fde01fd3380609884d181bf.jpg)

![b6da47b344286faf9e2f1af8817ca3bae095089ac8fd5c05b31ab6526768090f.jpg](../icml_results/714_Fixed%20Budget%20is%20No%20Harder%20Than%20Fixed%20Confidence%20in%20Best-Arm%20Identification%20up%20to%20Logarithmic%20Factors/tables/b6da47b344286faf9e2f1af8817ca3bae095089ac8fd5c05b31ab6526768090f.jpg)

<a id="paper-0705"></a>

## Dependence-Aware Label Aggregation for LLM-as-a-Judge via Ising Models


### Images

![181407f9b63a24c9f56ae176a95bad71966dbb96678e1ef6c276477c14fd848c.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/181407f9b63a24c9f56ae176a95bad71966dbb96678e1ef6c276477c14fd848c.jpg)

![3894c877cfbd73c1dc43ac4d9547b6b770f482e5b247aff92ed592dcf4c9347f.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/3894c877cfbd73c1dc43ac4d9547b6b770f482e5b247aff92ed592dcf4c9347f.jpg)

![72edd47f9a51117ada4523ff88f6b4235290630f53daf5d97c785feff651080d.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/72edd47f9a51117ada4523ff88f6b4235290630f53daf5d97c785feff651080d.jpg)

![7f8b7342f4572d22593875eca4eda73d19c01a0137836dd3f6783114974cdf3e.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/7f8b7342f4572d22593875eca4eda73d19c01a0137836dd3f6783114974cdf3e.jpg)

![97463139398fc0612a2d3e170db07a978f3aee0a494281d94469b68454c10de5.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/97463139398fc0612a2d3e170db07a978f3aee0a494281d94469b68454c10de5.jpg)

![bb6efe39a7693c04b94cb3b4aaf1a8d973cf25a0765b73744e67f82da4159a78.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/bb6efe39a7693c04b94cb3b4aaf1a8d973cf25a0765b73744e67f82da4159a78.jpg)

![d5c65f051033013879c9c0763c550e66d2162d427981cc4f9f5ef20285d6100d.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/d5c65f051033013879c9c0763c550e66d2162d427981cc4f9f5ef20285d6100d.jpg)

![fc5073a756b73024e91f8f8b9c7f5aa83680ebdb20a86d5f92a9ddfcad804c55.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/images/fc5073a756b73024e91f8f8b9c7f5aa83680ebdb20a86d5f92a9ddfcad804c55.jpg)

### Tables

![2ae85b7271e7ad9a5e2bcb482b485147aad81e9705aa1b79079f936b007ebe83.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/2ae85b7271e7ad9a5e2bcb482b485147aad81e9705aa1b79079f936b007ebe83.jpg)

![53ef4d6a1e6da32aedbba74899efbec47b021ba338ae455d7d5a78573e5b1206.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/53ef4d6a1e6da32aedbba74899efbec47b021ba338ae455d7d5a78573e5b1206.jpg)

![67d9a9a248372bfcac3ab63c338dfc038d1ac704eef5156f77d65db0677e8e5d.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/67d9a9a248372bfcac3ab63c338dfc038d1ac704eef5156f77d65db0677e8e5d.jpg)

![809e6f75ea645ca6a1da5986d61a8fc078dc32255ea914720f744ceefadf8465.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/809e6f75ea645ca6a1da5986d61a8fc078dc32255ea914720f744ceefadf8465.jpg)

![878238281252ebb012fc6fc15b3fd0ce1f7cf77c58124e5630201fb37c005714.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/878238281252ebb012fc6fc15b3fd0ce1f7cf77c58124e5630201fb37c005714.jpg)

![b0ed900eb7e2204d8ea29f989fa0066f2b7ed16bf362bc5de5180de5b6292f03.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/b0ed900eb7e2204d8ea29f989fa0066f2b7ed16bf362bc5de5180de5b6292f03.jpg)

![f71099e23158d626b25acd852e249169c91c5b9093bb1ab4afbeade314a450d0.jpg](../icml_results/715_Dependence-Aware%20Label%20Aggregation%20for%20LLM-as-a-Judge%20via%20Ising%20Models/tables/f71099e23158d626b25acd852e249169c91c5b9093bb1ab4afbeade314a450d0.jpg)

<a id="paper-0706"></a>

## Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics


### Images

![21081b5565057611ce9544347e696cf0a64ba407f26cb3f4b05bba96aa64627c.jpg](../icml_results/716_Dimension-Free%20Multimodal%20Sampling%20via%20Preconditioned%20Annealed%20Langevin%20Dynamics/images/21081b5565057611ce9544347e696cf0a64ba407f26cb3f4b05bba96aa64627c.jpg)

![610e0dafc61e87f9a404f94196a91e22ae8cc4259b297fcd6c28b8f833538f39.jpg](../icml_results/716_Dimension-Free%20Multimodal%20Sampling%20via%20Preconditioned%20Annealed%20Langevin%20Dynamics/images/610e0dafc61e87f9a404f94196a91e22ae8cc4259b297fcd6c28b8f833538f39.jpg)

![a2ec8c47748f021407b1dd3042dbb35a1f90faed7ea283c628c15695503d863c.jpg](../icml_results/716_Dimension-Free%20Multimodal%20Sampling%20via%20Preconditioned%20Annealed%20Langevin%20Dynamics/images/a2ec8c47748f021407b1dd3042dbb35a1f90faed7ea283c628c15695503d863c.jpg)

![a3f1590acee36bc36a17879d12bf33b821461f20ed89cf15dda234d324506e06.jpg](../icml_results/716_Dimension-Free%20Multimodal%20Sampling%20via%20Preconditioned%20Annealed%20Langevin%20Dynamics/images/a3f1590acee36bc36a17879d12bf33b821461f20ed89cf15dda234d324506e06.jpg)

![b4ce9d69aa142724c3e5969aa5b4364276aeafb0a67932fe1c7762ef5d2eab42.jpg](../icml_results/716_Dimension-Free%20Multimodal%20Sampling%20via%20Preconditioned%20Annealed%20Langevin%20Dynamics/images/b4ce9d69aa142724c3e5969aa5b4364276aeafb0a67932fe1c7762ef5d2eab42.jpg)

![bb61972f55359cc292e1309ed226de855a3f6391ffc1e3d5bfaaac45146e09a3.jpg](../icml_results/716_Dimension-Free%20Multimodal%20Sampling%20via%20Preconditioned%20Annealed%20Langevin%20Dynamics/images/bb61972f55359cc292e1309ed226de855a3f6391ffc1e3d5bfaaac45146e09a3.jpg)

<a id="paper-0707"></a>

## Regret-Based Federated Causal Discovery with Unknown Interventions


### Images

![073f367957947686df79ef187f970b500cbedd84e610996f20e634f6280eb3cb.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/073f367957947686df79ef187f970b500cbedd84e610996f20e634f6280eb3cb.jpg)

![371a89c606956085515338952ac5001736336f3cae6e0788ad6bfad19991bce0.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/371a89c606956085515338952ac5001736336f3cae6e0788ad6bfad19991bce0.jpg)

![6132a0e252d2714f9738f1874d6d9bf9043b471fc6e2ee6660c83a4f7775a73f.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/6132a0e252d2714f9738f1874d6d9bf9043b471fc6e2ee6660c83a4f7775a73f.jpg)

![74dbef27a6b744423b841743581adaa02c9482072bc5dbf0e529c562ed9f21b1.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/74dbef27a6b744423b841743581adaa02c9482072bc5dbf0e529c562ed9f21b1.jpg)

![98b303d47361673e97047192702f1c37da9583389374a074a041ca4fe50c0bf2.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/98b303d47361673e97047192702f1c37da9583389374a074a041ca4fe50c0bf2.jpg)

![9e0180feb22f2d9dd692e3860f5a51df813e5df898a37bb1c2733a7f0f537479.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/9e0180feb22f2d9dd692e3860f5a51df813e5df898a37bb1c2733a7f0f537479.jpg)

![a83567c5eae91414d74d4da85a0b94f4117c6ca6c801ddb8e02b29732959a050.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/a83567c5eae91414d74d4da85a0b94f4117c6ca6c801ddb8e02b29732959a050.jpg)

![c2398ea26b90e057b0c00b6c3b277cd39e2fead50f29f3d355fbbafdfad2f71b.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/c2398ea26b90e057b0c00b6c3b277cd39e2fead50f29f3d355fbbafdfad2f71b.jpg)

![cd8057db0d746b22edf8ea34a19dac14cfe513da747f0bd49f6e8f50845d8ebc.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/images/cd8057db0d746b22edf8ea34a19dac14cfe513da747f0bd49f6e8f50845d8ebc.jpg)

### Tables

![12f6c5a0e76174fb1674be730490e461d8612e4222be2fbf3a192dc2bfbeae2e.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/tables/12f6c5a0e76174fb1674be730490e461d8612e4222be2fbf3a192dc2bfbeae2e.jpg)

![46cc301a3dc3152d86ed3bdfdd0bf17c58ee8ebec71c84894e311a4f630559a6.jpg](../icml_results/717_Regret-Based%20Federated%20Causal%20Discovery%20with%20Unknown%20Interventions/tables/46cc301a3dc3152d86ed3bdfdd0bf17c58ee8ebec71c84894e311a4f630559a6.jpg)

<a id="paper-0708"></a>

## Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models


### Images

![05e44664c14755d4dffb331cb75676c7c4e1b23f5843ccd596b09315a728d3b3.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/05e44664c14755d4dffb331cb75676c7c4e1b23f5843ccd596b09315a728d3b3.jpg)

![061027fdb2ebb3939b03405d7539849c8d21709914328a14235801e209e989d1.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/061027fdb2ebb3939b03405d7539849c8d21709914328a14235801e209e989d1.jpg)

![0aff3403e4a38cc1d4a4088600e9cf2a42c6168ea9100c4352826a8783c37640.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/0aff3403e4a38cc1d4a4088600e9cf2a42c6168ea9100c4352826a8783c37640.jpg)

![0c04b27bb79f646158da4339c9a088a278723c202e3d9b141263d322d440b8aa.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/0c04b27bb79f646158da4339c9a088a278723c202e3d9b141263d322d440b8aa.jpg)

![11df22f527da2c32258550e5408cf2e6ff2f235a74bceb02bfcbf036d756e715.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/11df22f527da2c32258550e5408cf2e6ff2f235a74bceb02bfcbf036d756e715.jpg)

![189d5011a368c6ef06977accad451ec56938fa2f37e4dec8255de7bc7acbd5b4.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/189d5011a368c6ef06977accad451ec56938fa2f37e4dec8255de7bc7acbd5b4.jpg)

![1c3f24d5b79f740c52f157bdcd94d7328653410f1a129d7621ba59c524391000.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/1c3f24d5b79f740c52f157bdcd94d7328653410f1a129d7621ba59c524391000.jpg)

![207c526b8528346785be817991e9b24d5521f66f4984af52fdcb4b76717dd6d3.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/207c526b8528346785be817991e9b24d5521f66f4984af52fdcb4b76717dd6d3.jpg)

![246cb7f8d97e4fd42d34b74c6b54bad48bed710564b0222594997ffa695e425c.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/246cb7f8d97e4fd42d34b74c6b54bad48bed710564b0222594997ffa695e425c.jpg)

![2772178a1f2f8379766db7801a901993ae4be97dd7509f135e38c4379c73f0b9.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/2772178a1f2f8379766db7801a901993ae4be97dd7509f135e38c4379c73f0b9.jpg)

![28454502e4c9d5186931084d3944b0ab1123dcfd1a344473785ad20b127472f2.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/28454502e4c9d5186931084d3944b0ab1123dcfd1a344473785ad20b127472f2.jpg)

![2a959c461c21f75dc8c3b73122f1308f0c09f3720e81f7a64d7baf95539cb8be.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/2a959c461c21f75dc8c3b73122f1308f0c09f3720e81f7a64d7baf95539cb8be.jpg)

![2c4a3594f78a5c2bad29108a1d95a27499d7a75ace595d4f3eedbcb923279a18.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/2c4a3594f78a5c2bad29108a1d95a27499d7a75ace595d4f3eedbcb923279a18.jpg)

![3818213f7d11e191d5937aa6c8084400409c04ea4efb09c8a0d698080570a451.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/3818213f7d11e191d5937aa6c8084400409c04ea4efb09c8a0d698080570a451.jpg)

![38409a155b187162ab1324b3120d480056f5b7402bdc672e5ad2be17338c0f5e.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/38409a155b187162ab1324b3120d480056f5b7402bdc672e5ad2be17338c0f5e.jpg)

![384750da30e816e2126adc292629c722408800bab8a28ec1897514c8de190f8b.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/384750da30e816e2126adc292629c722408800bab8a28ec1897514c8de190f8b.jpg)

![399b7d6fddabd06a1cc6891723964bd14163029bd152a8c434636a6afbaace66.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/399b7d6fddabd06a1cc6891723964bd14163029bd152a8c434636a6afbaace66.jpg)

![39cc5d2df8896d7b8fe2fed96f4d9a4903116c895166fd7e8b239c67f4e5be4d.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/39cc5d2df8896d7b8fe2fed96f4d9a4903116c895166fd7e8b239c67f4e5be4d.jpg)

![3e4b0eb3ca347180df0b52e419fd52bb92a781d9e044e35ab186db4f98d9c589.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/3e4b0eb3ca347180df0b52e419fd52bb92a781d9e044e35ab186db4f98d9c589.jpg)

![3eac648368ab14b2e7e1056ef290bd9e8db42b0ee4e94559ed863f6dab307dd7.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/3eac648368ab14b2e7e1056ef290bd9e8db42b0ee4e94559ed863f6dab307dd7.jpg)

![3fd76a30c01ce2aa52551119e4d22bc34b92aa4d4bbf916ee3d61bef38cea5ee.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/3fd76a30c01ce2aa52551119e4d22bc34b92aa4d4bbf916ee3d61bef38cea5ee.jpg)

![429eb88d719c1c9505e87431d7bb06ddb60253a901035c8652b44127bd9b6ac4.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/429eb88d719c1c9505e87431d7bb06ddb60253a901035c8652b44127bd9b6ac4.jpg)

![45b426cf3aaf1fce306e6f1482dbe3c811fae9b7b8ab3e476a72bfebc097d09c.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/45b426cf3aaf1fce306e6f1482dbe3c811fae9b7b8ab3e476a72bfebc097d09c.jpg)

![4997345100b9593302ed125f791be36727fe496ff4fa23dcc249acebc7da1503.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/4997345100b9593302ed125f791be36727fe496ff4fa23dcc249acebc7da1503.jpg)

![4b02aee925df723bda98da7633e975d24413ecd1effaa62232fb251618c70348.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/4b02aee925df723bda98da7633e975d24413ecd1effaa62232fb251618c70348.jpg)

![4c29b0f0e580cc4ee4bf724285bdfe0ca71932c9ced8ea7564b7a3f85ec3d479.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/4c29b0f0e580cc4ee4bf724285bdfe0ca71932c9ced8ea7564b7a3f85ec3d479.jpg)

![4feb5a3f30acffbce50e1feddc24340aa4ac981fb95fab9443fb4d59ba7591f9.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/4feb5a3f30acffbce50e1feddc24340aa4ac981fb95fab9443fb4d59ba7591f9.jpg)

![58e9f8c949545a668d0683ff007e7e8f3d60fb834fe185194d98e5f91ef8831a.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/58e9f8c949545a668d0683ff007e7e8f3d60fb834fe185194d98e5f91ef8831a.jpg)

![5c3becb84f1add48ac9c44ba8e5bc5af18693ebbcef44f53ae1c53bc6d3dc505.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/5c3becb84f1add48ac9c44ba8e5bc5af18693ebbcef44f53ae1c53bc6d3dc505.jpg)

![5db5c90ec22816caea2f8b6949f8cf9fcebdc8fb82d16805e0c7ef4a0f9887ff.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/5db5c90ec22816caea2f8b6949f8cf9fcebdc8fb82d16805e0c7ef4a0f9887ff.jpg)

![62de90bf3e514dbc1d24e93c0d08b624c0e474f33ec23718c625c7113f5bd1f9.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/62de90bf3e514dbc1d24e93c0d08b624c0e474f33ec23718c625c7113f5bd1f9.jpg)

![6c0d0ec575fb918da767ad015ae95ea5216f88c49265418db7de66fca20e32c4.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/6c0d0ec575fb918da767ad015ae95ea5216f88c49265418db7de66fca20e32c4.jpg)

![7e346520d2edc6cc14acb1b8b3cd897bfa0a9a26d71bdc2c1f05c63e4b574a18.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/7e346520d2edc6cc14acb1b8b3cd897bfa0a9a26d71bdc2c1f05c63e4b574a18.jpg)

![82a824fbd7dd4eb5037ff6fddb9286773598560d1411416faead19873dbcfc49.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/82a824fbd7dd4eb5037ff6fddb9286773598560d1411416faead19873dbcfc49.jpg)

![96b9fcacd7d86222697a6013947d3eb97f41b749a050a28e2cc0eb5180689e25.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/96b9fcacd7d86222697a6013947d3eb97f41b749a050a28e2cc0eb5180689e25.jpg)

![9a11b672f2f0dd493ad6af85ba8b3fbe3f72026947fb4c94ade7399ecd62134a.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/9a11b672f2f0dd493ad6af85ba8b3fbe3f72026947fb4c94ade7399ecd62134a.jpg)

![9c0dfe9d7cabb11a9049113ffe38cae6830ec7d6efedb2028c705c486718bbb2.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/9c0dfe9d7cabb11a9049113ffe38cae6830ec7d6efedb2028c705c486718bbb2.jpg)

![9eec22b760c28f3be39f0e82b498ea49b02d8b5a44e188095b60990f3431684a.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/9eec22b760c28f3be39f0e82b498ea49b02d8b5a44e188095b60990f3431684a.jpg)

![a07b56608e39383cf7cfab8a315faf60c577c8ae583f2432774e47972ef225a1.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/a07b56608e39383cf7cfab8a315faf60c577c8ae583f2432774e47972ef225a1.jpg)

![a8f9ee38c22e5a353fb34b592b8686e8265a6ec29cca3dfea7aa02e19d47fd73.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/a8f9ee38c22e5a353fb34b592b8686e8265a6ec29cca3dfea7aa02e19d47fd73.jpg)

![aad5f22b019d7755ea8bc9036112a0f4b5d4c565700c2b810fb14189d3638d27.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/aad5f22b019d7755ea8bc9036112a0f4b5d4c565700c2b810fb14189d3638d27.jpg)

![ac9da63f646bbb95d0f78a1d0c58d93d19a0d8a73c026846626a2128635f9231.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/ac9da63f646bbb95d0f78a1d0c58d93d19a0d8a73c026846626a2128635f9231.jpg)

![b4c374476a40018a5b7e4027ea82f1a9ae50b7c74eeb2273bdd6947b20a4a673.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/b4c374476a40018a5b7e4027ea82f1a9ae50b7c74eeb2273bdd6947b20a4a673.jpg)

![b50463353d1e95abb5d75071b938264bbef5128448b26467755f217d7451aa28.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/b50463353d1e95abb5d75071b938264bbef5128448b26467755f217d7451aa28.jpg)

![b76793af50e0cc18ec1cd70b4dccd78fb3d3265253168ecad3df894dc4d56997.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/b76793af50e0cc18ec1cd70b4dccd78fb3d3265253168ecad3df894dc4d56997.jpg)

![b8fa7fd07efadd60fa5ee28d034e56edf0653fface1fe3383edbfb8c4e6d173a.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/b8fa7fd07efadd60fa5ee28d034e56edf0653fface1fe3383edbfb8c4e6d173a.jpg)

![bd1245d4bb5a36430439328967554f652aca25d15dfc7e02542ff3cedfe81fef.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/bd1245d4bb5a36430439328967554f652aca25d15dfc7e02542ff3cedfe81fef.jpg)

![bdde28154752d59ee8bb432d3f8cbc7b26374800f4cebed6df359e47f70e069c.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/bdde28154752d59ee8bb432d3f8cbc7b26374800f4cebed6df359e47f70e069c.jpg)

![c06bf9615fc20cf6f0939d02405b858f417f5fb46005c661441370bb3835c693.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/c06bf9615fc20cf6f0939d02405b858f417f5fb46005c661441370bb3835c693.jpg)

![ca8b98e097492143bcb34bc1c5586963d6c3573b0525a30acb2844d1382b10a3.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/ca8b98e097492143bcb34bc1c5586963d6c3573b0525a30acb2844d1382b10a3.jpg)

![cd9809dd60319f173d8c4ead95288a0af0618b3cd823b18d489c9b286a0307da.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/cd9809dd60319f173d8c4ead95288a0af0618b3cd823b18d489c9b286a0307da.jpg)

![d8b9556436e270cfb24450451017533e22d1fe084920a614e5403c833633a571.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/d8b9556436e270cfb24450451017533e22d1fe084920a614e5403c833633a571.jpg)

![df1aae12167875db7221bfeb3b53ef25e52c7fa25c6bdc074216fc0ce49b5a8e.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/df1aae12167875db7221bfeb3b53ef25e52c7fa25c6bdc074216fc0ce49b5a8e.jpg)

![e1969eb606d7467d02ad10f3733bea8c5e9004a5dc829dbecd54a7dd66489162.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/e1969eb606d7467d02ad10f3733bea8c5e9004a5dc829dbecd54a7dd66489162.jpg)

![eac9640a6e0f121c366b59e90a28879cd84515c1b99d2d0e9deca69f8e36888f.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/eac9640a6e0f121c366b59e90a28879cd84515c1b99d2d0e9deca69f8e36888f.jpg)

![f37edf676134255bb0d96d680bc6aa07c2e0f066974580bf0a7df41bb4bedf01.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/f37edf676134255bb0d96d680bc6aa07c2e0f066974580bf0a7df41bb4bedf01.jpg)

![f592abeedd622ba4c84f62d599bf168d823811e2bc4a8579dcc89cdab169864b.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/f592abeedd622ba4c84f62d599bf168d823811e2bc4a8579dcc89cdab169864b.jpg)

![ff6911ab6b99b5ea60577315631506ba1f96c489786ad2625e43067240defddd.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/images/ff6911ab6b99b5ea60577315631506ba1f96c489786ad2625e43067240defddd.jpg)

### Tables

![17c0b8644f572ac8533f5e8b79454b5ab39668f89546affb1ddaac31b9707a02.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/17c0b8644f572ac8533f5e8b79454b5ab39668f89546affb1ddaac31b9707a02.jpg)

![1bf23e7bbfac884219010d594077e290c4591f1796bc2d359f8a3baa5fa1d74f.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/1bf23e7bbfac884219010d594077e290c4591f1796bc2d359f8a3baa5fa1d74f.jpg)

![32312ee8242a20f13edd908b55decb40b23f82fc3e87007f46697229b10ee149.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/32312ee8242a20f13edd908b55decb40b23f82fc3e87007f46697229b10ee149.jpg)

![537201f6e3d32ab0d4565524feaaa9df97fcac7b8008ae9b677bb4c4026137e9.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/537201f6e3d32ab0d4565524feaaa9df97fcac7b8008ae9b677bb4c4026137e9.jpg)

![59e8196660701809d36ea70e8c70eca8f2e35dc8076ffdbf5dc0c54989995654.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/59e8196660701809d36ea70e8c70eca8f2e35dc8076ffdbf5dc0c54989995654.jpg)

![6b3c451e918193a724f7f754e9f3e5957b5e1f446f7e19d74aa080ee763b9d83.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/6b3c451e918193a724f7f754e9f3e5957b5e1f446f7e19d74aa080ee763b9d83.jpg)

![75e5ec9d99db950f975b2e71f7c7311349c2f31d3411011797952601e29b7cda.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/75e5ec9d99db950f975b2e71f7c7311349c2f31d3411011797952601e29b7cda.jpg)

![78e76ecc338dafba0b114f035701c45f5b1bf54eda6d293cf10a6e7820ad4cb1.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/78e76ecc338dafba0b114f035701c45f5b1bf54eda6d293cf10a6e7820ad4cb1.jpg)

![841d50719ad79b0b063a396950c617c7c33cc47f1b589efce00df3ec2dfc46fc.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/841d50719ad79b0b063a396950c617c7c33cc47f1b589efce00df3ec2dfc46fc.jpg)

![89d561cd49a5936277c1ddb3a43a9fa3b5687a5fb37b009c8611526e61dbab0c.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/89d561cd49a5936277c1ddb3a43a9fa3b5687a5fb37b009c8611526e61dbab0c.jpg)

![8af2a2c6a8eb4d54c0113310d6381c16df7e334b95c6facbad82ef0bf5b3ce87.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/8af2a2c6a8eb4d54c0113310d6381c16df7e334b95c6facbad82ef0bf5b3ce87.jpg)

![92419849ca2fc797db5e3a6042f1bca47d9faa63726f413681356f3b1d8e7152.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/92419849ca2fc797db5e3a6042f1bca47d9faa63726f413681356f3b1d8e7152.jpg)

![a7238eedc24cab8eea4291ee9f5ee2bb61a1da903f9a73d3956d91b48979415b.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/a7238eedc24cab8eea4291ee9f5ee2bb61a1da903f9a73d3956d91b48979415b.jpg)

![a7f2c359d60c87a0e293ddae8281cbc7efce1dc36e85584198dcde9627d94b63.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/a7f2c359d60c87a0e293ddae8281cbc7efce1dc36e85584198dcde9627d94b63.jpg)

![adba8f09f8cd8badce46f24d1323f7997398d33cc7a4a0504bf9ffda4c439cf3.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/adba8f09f8cd8badce46f24d1323f7997398d33cc7a4a0504bf9ffda4c439cf3.jpg)

![e68fbf8bbd49fc4a40f7a89caecde96a4df48c324dfc9036146574371a951a61.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/e68fbf8bbd49fc4a40f7a89caecde96a4df48c324dfc9036146574371a951a61.jpg)

![f87d900f74a42d7ea9d9b94071c9f45f1ed2340ff2d41787fd120f3f5087de9b.jpg](../icml_results/718_Is%20One%20Layer%20Enough_%20Understanding%20Inference%20Dynamics%20in%20Tabular%20Foundation%20Models/tables/f87d900f74a42d7ea9d9b94071c9f45f1ed2340ff2d41787fd120f3f5087de9b.jpg)

<a id="paper-0709"></a>

## Quantifying LLM Attention-Head Stability: Implications for Circuit Universality


### Images

![0030157459803633ac868e114c462a1515ceba343c2ce17f25c6a712e4cf2881.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/0030157459803633ac868e114c462a1515ceba343c2ce17f25c6a712e4cf2881.jpg)

![00e4d11fbaf417be1120ab39634d49ff81ee634c6d1d35ca0d9b6b9bfc049ddb.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/00e4d11fbaf417be1120ab39634d49ff81ee634c6d1d35ca0d9b6b9bfc049ddb.jpg)

![024efa0535689b8557c707ae85dff7bef410cea60c2f0d8fcea5f9b080b26d80.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/024efa0535689b8557c707ae85dff7bef410cea60c2f0d8fcea5f9b080b26d80.jpg)

![06c5a1e2fab2ec4344273d610055386efdbd17ed5a00d2f61983e43113f7c8bd.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/06c5a1e2fab2ec4344273d610055386efdbd17ed5a00d2f61983e43113f7c8bd.jpg)

![0820d885ad5f4321df6916812ce36d225d52ded269858fe9ceedac508bf33bec.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/0820d885ad5f4321df6916812ce36d225d52ded269858fe9ceedac508bf33bec.jpg)

![0880b3ed49b3a2f4a5d5ffff0e6b1f6b002a0682f684a6e16bd61645946164dc.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/0880b3ed49b3a2f4a5d5ffff0e6b1f6b002a0682f684a6e16bd61645946164dc.jpg)

![092bb3000e2571e2425321b4086a85792a24d4ffb3178ae0ce8c31fc6af28779.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/092bb3000e2571e2425321b4086a85792a24d4ffb3178ae0ce8c31fc6af28779.jpg)

![0d2494f66ad432ef8f3e8b2fd490a0fc84758f5a73003f086872357f758deb56.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/0d2494f66ad432ef8f3e8b2fd490a0fc84758f5a73003f086872357f758deb56.jpg)

![0f41714a820ccbd7c4464f7da4f2fc6058d67f681f07a8bf2956c135a6ed2f4c.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/0f41714a820ccbd7c4464f7da4f2fc6058d67f681f07a8bf2956c135a6ed2f4c.jpg)

![109c18bccf883c631e0c18d39427499306454c9651a0e9e98a73595b56c0ddce.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/109c18bccf883c631e0c18d39427499306454c9651a0e9e98a73595b56c0ddce.jpg)

![11e860e9240e2a704631de5144fef52c030d66158f46ed0a995a467dc845a8a8.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/11e860e9240e2a704631de5144fef52c030d66158f46ed0a995a467dc845a8a8.jpg)

![14c35e0e1ebbd404c89190f14a0c735360e99cf6f1c43fc582d231e17ac3e6eb.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/14c35e0e1ebbd404c89190f14a0c735360e99cf6f1c43fc582d231e17ac3e6eb.jpg)

![17d58f0309db4fe7a9e83080392971bc597986504e53198fcbf4dc655099cc5b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/17d58f0309db4fe7a9e83080392971bc597986504e53198fcbf4dc655099cc5b.jpg)

![196c98112d21f4777504f0c4bcb447499bd65fdb21f02081bdeee2e66968dba6.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/196c98112d21f4777504f0c4bcb447499bd65fdb21f02081bdeee2e66968dba6.jpg)

![1a62d199724c4e2d5dae05fce498d6514bb4858fc7e41269bdc712e2895a9d2a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/1a62d199724c4e2d5dae05fce498d6514bb4858fc7e41269bdc712e2895a9d2a.jpg)

![1b3db6c44d6eab995bf433072144c56edb0913a551691c193c3dbec7e001bbbb.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/1b3db6c44d6eab995bf433072144c56edb0913a551691c193c3dbec7e001bbbb.jpg)

![1f1b3ef497bc68d5dd23ea52fc15a798a65b36bdf2cf00515ddb1d972c9f71f7.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/1f1b3ef497bc68d5dd23ea52fc15a798a65b36bdf2cf00515ddb1d972c9f71f7.jpg)

![212ab7c3e436962314c4c325fd96d8e656e9912d1559108914739f420950be00.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/212ab7c3e436962314c4c325fd96d8e656e9912d1559108914739f420950be00.jpg)

![24534b1cb9fa7a1f85a649bb5c48f5bab14934aa2528274832d493c3fbef9bcf.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/24534b1cb9fa7a1f85a649bb5c48f5bab14934aa2528274832d493c3fbef9bcf.jpg)

![24b0d95d14d46685739665f3cb5589a4038e9f0adf9057a7921290917200beb7.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/24b0d95d14d46685739665f3cb5589a4038e9f0adf9057a7921290917200beb7.jpg)

![28772f33e3734999668b35f0f75eae321f38cc7b2d3ae434eb6ed66db163448e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/28772f33e3734999668b35f0f75eae321f38cc7b2d3ae434eb6ed66db163448e.jpg)

![2b2ccf663fdc4fc7f3c48403e9de7845745f1484354e034503015ffb8918b949.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/2b2ccf663fdc4fc7f3c48403e9de7845745f1484354e034503015ffb8918b949.jpg)

![2baceb196edbf26c7085da913c0f938184fb137bb83cab4a2add1538dc5b403a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/2baceb196edbf26c7085da913c0f938184fb137bb83cab4a2add1538dc5b403a.jpg)

![2cca0e289054af00d57127ed23640af3df7ea5f07fe267fe7417146361d5c6f1.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/2cca0e289054af00d57127ed23640af3df7ea5f07fe267fe7417146361d5c6f1.jpg)

![2d9e710fe0fea8518692cd45d50824539f5a0d31551a5a085e2a647585eaa8f5.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/2d9e710fe0fea8518692cd45d50824539f5a0d31551a5a085e2a647585eaa8f5.jpg)

![2f4ae47d4e9e0e7671075ecaadd971c35e147621255fa73337e77c48819982ea.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/2f4ae47d4e9e0e7671075ecaadd971c35e147621255fa73337e77c48819982ea.jpg)

![319d56a098bc6841bca6ac84b50f9c427bfbf537f267c0838e9d49aae5c92a2b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/319d56a098bc6841bca6ac84b50f9c427bfbf537f267c0838e9d49aae5c92a2b.jpg)

![32249e7fad32e43e9f08009494ee17397cdf3881005e576949b4a09e77797957.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/32249e7fad32e43e9f08009494ee17397cdf3881005e576949b4a09e77797957.jpg)

![34d1a7e86e0874a1aa76285d24533e7dde01d938b3382f83b7774fc18750eb4e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/34d1a7e86e0874a1aa76285d24533e7dde01d938b3382f83b7774fc18750eb4e.jpg)

![368998ddfda38faf61b04aa556e2311416d99760c736b3305a8fbf706bcdac14.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/368998ddfda38faf61b04aa556e2311416d99760c736b3305a8fbf706bcdac14.jpg)

![399488ad4fb63b4b63ac85b84bb9942d9f0ba435b26a8416472e60d380d7abe6.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/399488ad4fb63b4b63ac85b84bb9942d9f0ba435b26a8416472e60d380d7abe6.jpg)

![3ba517ee690b344222424a82415b7666b3488547030b43e195c4c5eec7dc8cff.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/3ba517ee690b344222424a82415b7666b3488547030b43e195c4c5eec7dc8cff.jpg)

![3e1345f5095b860c630654e7c9ce71c8690aa927cac4618e1330e774b63da2c3.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/3e1345f5095b860c630654e7c9ce71c8690aa927cac4618e1330e774b63da2c3.jpg)

![3e466bb463793ff803edbe4bbd601a7a07465ae0220ade1faaa559a54059f4c2.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/3e466bb463793ff803edbe4bbd601a7a07465ae0220ade1faaa559a54059f4c2.jpg)

![3eb2c50a19b213b1e817614461e8236a741d9cff178eccc1071f2b5bfd0894de.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/3eb2c50a19b213b1e817614461e8236a741d9cff178eccc1071f2b5bfd0894de.jpg)

![3f6af73293c4ea3e47505cc1c909cf8b96edbba633d2c0a7cf06cecf24c6f958.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/3f6af73293c4ea3e47505cc1c909cf8b96edbba633d2c0a7cf06cecf24c6f958.jpg)

![419a037fcd8b945f880e085ec46f9ea0fd5bbadedff931f3d335827a5e33518e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/419a037fcd8b945f880e085ec46f9ea0fd5bbadedff931f3d335827a5e33518e.jpg)

![428f6bff4610761a85dea3a41752561580246b44fcdd963c2f9c6de1e72e1c4d.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/428f6bff4610761a85dea3a41752561580246b44fcdd963c2f9c6de1e72e1c4d.jpg)

![432b2d77b0ea3b039c53ee69044f1ebb0bb1e00056daeb7a8973c589fbd7a40a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/432b2d77b0ea3b039c53ee69044f1ebb0bb1e00056daeb7a8973c589fbd7a40a.jpg)

![44c648fa6cbf38385bc6ad3d971d608fb768f4f45e5bc1758cca49e9d2637960.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/44c648fa6cbf38385bc6ad3d971d608fb768f4f45e5bc1758cca49e9d2637960.jpg)

![45b1b6c0c8d57be6de81387d9004b4c9df87a6466d71948d6622f289d8fd1f0e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/45b1b6c0c8d57be6de81387d9004b4c9df87a6466d71948d6622f289d8fd1f0e.jpg)

![474e15d1f95d342226dcfd8201fa32c3f1070a43408f626f9da3b71d54939dd6.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/474e15d1f95d342226dcfd8201fa32c3f1070a43408f626f9da3b71d54939dd6.jpg)

![47c27a37763ad9f49966d8974225915c067b533c4aad89db887961efce656794.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/47c27a37763ad9f49966d8974225915c067b533c4aad89db887961efce656794.jpg)

![489d7ef86d60f96154a83ad810b214d2fe3e819144e69f6d0dc44d141fb2508d.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/489d7ef86d60f96154a83ad810b214d2fe3e819144e69f6d0dc44d141fb2508d.jpg)

![49b0d6f9572994c60468b70bf3e6df4f6f0680ed88081a583646f7f12e292652.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/49b0d6f9572994c60468b70bf3e6df4f6f0680ed88081a583646f7f12e292652.jpg)

![4d80a02b75760762df10597ca9a927ce842fc5a89624e7cb85db1fe59b531141.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/4d80a02b75760762df10597ca9a927ce842fc5a89624e7cb85db1fe59b531141.jpg)

![4df7208790d8720ed5302d592ddb23eae9847b0d8fb5fcb7a7e643cfc221231e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/4df7208790d8720ed5302d592ddb23eae9847b0d8fb5fcb7a7e643cfc221231e.jpg)

![4e0577d1443a1bc5514abff867d224198c0d83aa180529d876fbc808c3089e3b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/4e0577d1443a1bc5514abff867d224198c0d83aa180529d876fbc808c3089e3b.jpg)

![4fb2f7f6e875fbf7f9374752f722684cc4b276851cb519855a2ef27842064a3a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/4fb2f7f6e875fbf7f9374752f722684cc4b276851cb519855a2ef27842064a3a.jpg)

![5501c482c7e29aed32661f76de24da12bcd721f118f8cc14feadedbcbf2ea74d.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/5501c482c7e29aed32661f76de24da12bcd721f118f8cc14feadedbcbf2ea74d.jpg)

![59729be4cfdbe35899aef655fc142e5db31d0ce50ab737be0e195c7c8cd00276.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/59729be4cfdbe35899aef655fc142e5db31d0ce50ab737be0e195c7c8cd00276.jpg)

![5ac92f862839f88faafc4bdf8b517a38e20629a2862c533a1fc40b4c1da7bd80.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/5ac92f862839f88faafc4bdf8b517a38e20629a2862c533a1fc40b4c1da7bd80.jpg)

![5c0aa569f7660e69aef8ba8a4f86b122ccdf7a5f268e1028dabf1f8adb462802.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/5c0aa569f7660e69aef8ba8a4f86b122ccdf7a5f268e1028dabf1f8adb462802.jpg)

![62197d0c6e0895fddbbde7555322907c14dd1135b071a3fe64d422981b610014.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/62197d0c6e0895fddbbde7555322907c14dd1135b071a3fe64d422981b610014.jpg)

![67392cbddbd677beeb2cefd27938c2a08558d0b88593181c632a8f839b31a015.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/67392cbddbd677beeb2cefd27938c2a08558d0b88593181c632a8f839b31a015.jpg)

![69a02a6746413e8a427264584494c2d87428cc3c9f378a60271c8687743dc18b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/69a02a6746413e8a427264584494c2d87428cc3c9f378a60271c8687743dc18b.jpg)

![6d20d8b1255a83eb10c7f5ebeeb25fd9d3374bd16e2b6f1c0d2beb266cd22934.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/6d20d8b1255a83eb10c7f5ebeeb25fd9d3374bd16e2b6f1c0d2beb266cd22934.jpg)

![710d9571a71e3a81e3754e6f23142e411abeced37506e2ca06a9ec9a724a8b8a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/710d9571a71e3a81e3754e6f23142e411abeced37506e2ca06a9ec9a724a8b8a.jpg)

![7283a2ec9b5d8032e31f314eef594745842bd015d6d453b6db0acb2ca5166034.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/7283a2ec9b5d8032e31f314eef594745842bd015d6d453b6db0acb2ca5166034.jpg)

![72f6ead0ae0464a10a6beccb8992576ddd0e69de3cddc3d8443dfad9cc208d92.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/72f6ead0ae0464a10a6beccb8992576ddd0e69de3cddc3d8443dfad9cc208d92.jpg)

![7528ea6e91513f7b2abe326d1aebf5328822b9ccd0ad869eee1c68948284259c.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/7528ea6e91513f7b2abe326d1aebf5328822b9ccd0ad869eee1c68948284259c.jpg)

![7acbd2527f33e19657d80ccde5908d9bb3f470783085f7ee581a20c7ac7ac59a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/7acbd2527f33e19657d80ccde5908d9bb3f470783085f7ee581a20c7ac7ac59a.jpg)

![7bf1375b34618f3490514f22cfcd0e1c4ad315079c485f30ac49806d6d50786e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/7bf1375b34618f3490514f22cfcd0e1c4ad315079c485f30ac49806d6d50786e.jpg)

![7ce9a74397a0127cdd7f29b5141bf5cd35b4c365d419bb1ee775353e8a32504d.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/7ce9a74397a0127cdd7f29b5141bf5cd35b4c365d419bb1ee775353e8a32504d.jpg)

![7fbce13e5005c84d54f494a155e5dad944533798ca49f2f8596be144ebbe4778.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/7fbce13e5005c84d54f494a155e5dad944533798ca49f2f8596be144ebbe4778.jpg)

![801d545010cbca7afec8247082f75326c8d27216e42ee6b3ef74b0de9757bca7.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/801d545010cbca7afec8247082f75326c8d27216e42ee6b3ef74b0de9757bca7.jpg)

![83b95fbb1c6084a7f2046bc683dbb4aa5b20c61330c723b757b38e8f7c93f2ef.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/83b95fbb1c6084a7f2046bc683dbb4aa5b20c61330c723b757b38e8f7c93f2ef.jpg)

![848bf583b97eacda798feca6a7aa616dbc1d50f5ad5b22d01961ea98ee54fa9c.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/848bf583b97eacda798feca6a7aa616dbc1d50f5ad5b22d01961ea98ee54fa9c.jpg)

![85d691764dedfbecb6853d40b1ba0c7627c42cab9e0818cbf746060e3dcb01d0.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/85d691764dedfbecb6853d40b1ba0c7627c42cab9e0818cbf746060e3dcb01d0.jpg)

![85e17819599e14aec5b31eeb4587494db40f50044fd4b99a7b16af3ccf2f8d12.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/85e17819599e14aec5b31eeb4587494db40f50044fd4b99a7b16af3ccf2f8d12.jpg)

![874933c982f46ef8fd0217c24c239f2227715b1c474d7fc9cc437a94fb254c59.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/874933c982f46ef8fd0217c24c239f2227715b1c474d7fc9cc437a94fb254c59.jpg)

![88edf366daaf677018db253199bb9714d070b7d9cf9d140c42c56834718fc195.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/88edf366daaf677018db253199bb9714d070b7d9cf9d140c42c56834718fc195.jpg)

![911470f0248bbc0d70ee2a01128b2f94cc24620c780ad36f1027603dda8b8b63.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/911470f0248bbc0d70ee2a01128b2f94cc24620c780ad36f1027603dda8b8b63.jpg)

![9885795fc61098ece20081de4d7d13b524a83e97adf38129be9922245debcf31.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/9885795fc61098ece20081de4d7d13b524a83e97adf38129be9922245debcf31.jpg)

![9ae3872da415c49f3c174477c6929d0fec15ed896ba5d7b799f1ba62f203c377.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/9ae3872da415c49f3c174477c6929d0fec15ed896ba5d7b799f1ba62f203c377.jpg)

![9cb401c40be163f81ce640d71d02e8a21d105d7913c605723a32672ad02f1ed7.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/9cb401c40be163f81ce640d71d02e8a21d105d7913c605723a32672ad02f1ed7.jpg)

![9e5ea11aaba8a517fd2c62d1e30687b04cf0c63978bc68f313e48839f93604c2.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/9e5ea11aaba8a517fd2c62d1e30687b04cf0c63978bc68f313e48839f93604c2.jpg)

![9ecf7921ea91e74a1d972bc4a37c525b8cdcde89d184296fc9996e5326cd4554.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/9ecf7921ea91e74a1d972bc4a37c525b8cdcde89d184296fc9996e5326cd4554.jpg)

![a079d9bb5f989691d180841c738a73d8d2cc7975b564a5e35de67c16599fdb8f.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/a079d9bb5f989691d180841c738a73d8d2cc7975b564a5e35de67c16599fdb8f.jpg)

![a2d9689fe816bdd9531d793bd9ac37144bb0209f783d4e8710e0c750dc9cf258.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/a2d9689fe816bdd9531d793bd9ac37144bb0209f783d4e8710e0c750dc9cf258.jpg)

![a38c4b4ac94c89038437dedffd17bffcf7f22d30736a7078a3425c7000a815ae.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/a38c4b4ac94c89038437dedffd17bffcf7f22d30736a7078a3425c7000a815ae.jpg)

![a5ff058ddfa9929fcc6fd0f8910b2e832ce90a521103295ec6e49850f685d4a3.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/a5ff058ddfa9929fcc6fd0f8910b2e832ce90a521103295ec6e49850f685d4a3.jpg)

![a791f7628f423647c55c0f05d21d2f245f9307013742377905046145c0523a8e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/a791f7628f423647c55c0f05d21d2f245f9307013742377905046145c0523a8e.jpg)

![a8b375a4c47e4c85649475effdfae68a6912e9cdcdb6d3da2fbf49664dffdd5b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/a8b375a4c47e4c85649475effdfae68a6912e9cdcdb6d3da2fbf49664dffdd5b.jpg)

![ad949a820af373b7cb7ca88be661046f7b974bb73b716f7f65394f3d618c78c6.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/ad949a820af373b7cb7ca88be661046f7b974bb73b716f7f65394f3d618c78c6.jpg)

![aef8ba3a50d47fb070f4c1766a7be7b0c2aa5ce7cd52e92ca86fc623f6cf1fec.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/aef8ba3a50d47fb070f4c1766a7be7b0c2aa5ce7cd52e92ca86fc623f6cf1fec.jpg)

![affd9abcc8d34aab4fd28c9b85762ed93770a2699e5f30f2cdefc6bc637e79e1.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/affd9abcc8d34aab4fd28c9b85762ed93770a2699e5f30f2cdefc6bc637e79e1.jpg)

![b14c95bb667fa3f614b687b9a3e6e1840325149143a2a973b3df2c4611b38fc5.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/b14c95bb667fa3f614b687b9a3e6e1840325149143a2a973b3df2c4611b38fc5.jpg)

![b50b4a9863459b830cd73740c768f0fecf505ec60dac9f08c25d7bf5b8bb937b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/b50b4a9863459b830cd73740c768f0fecf505ec60dac9f08c25d7bf5b8bb937b.jpg)

![b53784a37bced8d0de6190bf14adc8ee5eeb0b8feea4877ff3fbdbf18699f048.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/b53784a37bced8d0de6190bf14adc8ee5eeb0b8feea4877ff3fbdbf18699f048.jpg)

![b5759469f2f9a6e26676460734838b65461c6141cd1a022f2208a764b64f6167.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/b5759469f2f9a6e26676460734838b65461c6141cd1a022f2208a764b64f6167.jpg)

![b6a7751dc142ac8366cbc581a72945afe869fe0439e61ed7e5b2352368e85fe0.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/b6a7751dc142ac8366cbc581a72945afe869fe0439e61ed7e5b2352368e85fe0.jpg)

![b77ccfbb1221524a508cd391f88b4b0592c3216d21cff4b42e8a5dc96d87ca3d.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/b77ccfbb1221524a508cd391f88b4b0592c3216d21cff4b42e8a5dc96d87ca3d.jpg)

![ba66ccc7bd202ccbb3cc4b8b5f7f3350ba38ebfc478b566f78171d73f8e03f66.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/ba66ccc7bd202ccbb3cc4b8b5f7f3350ba38ebfc478b566f78171d73f8e03f66.jpg)

![bd1350267f69c2496aa3d6315cb4a9ffeb9278a8543ab0f9cd297c1f8e8ea7f6.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/bd1350267f69c2496aa3d6315cb4a9ffeb9278a8543ab0f9cd297c1f8e8ea7f6.jpg)

![bd93ab595a548cc6e8d660d02438c410aab48ea2615cbda213632b3d8f7dc580.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/bd93ab595a548cc6e8d660d02438c410aab48ea2615cbda213632b3d8f7dc580.jpg)

![bde36b161d32e95d59726f6646718465998c1693d9ba1b3c02ee789547d15f35.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/bde36b161d32e95d59726f6646718465998c1693d9ba1b3c02ee789547d15f35.jpg)

![c1761e679dd346ea6d03a6172eb3aa632b49924db4b2e7d4c35366d7d4c26a2a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/c1761e679dd346ea6d03a6172eb3aa632b49924db4b2e7d4c35366d7d4c26a2a.jpg)

![c40bc76c8a3fb32e976eb899da186f91146f1358dbd289b105a9a737850f7b3e.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/c40bc76c8a3fb32e976eb899da186f91146f1358dbd289b105a9a737850f7b3e.jpg)

![c41849a4f605f9a3f8c3b594f4ee24337634d253b7815c2e2feea7ea7e3089ec.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/c41849a4f605f9a3f8c3b594f4ee24337634d253b7815c2e2feea7ea7e3089ec.jpg)

![c724d8a43a8f4510727992c7949a6ee5803fe2a2edd27feb1bdfe5be7a6609f0.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/c724d8a43a8f4510727992c7949a6ee5803fe2a2edd27feb1bdfe5be7a6609f0.jpg)

![c7be45623e410f83305e61c370098265feb3a6c94dba3ef9003d35ae3ce64709.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/c7be45623e410f83305e61c370098265feb3a6c94dba3ef9003d35ae3ce64709.jpg)

![cbab69cbb5118645399ec971ba81a2286ca78b3f8496ca9f14bed585fab0e326.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/cbab69cbb5118645399ec971ba81a2286ca78b3f8496ca9f14bed585fab0e326.jpg)

![cc40432afd0554c723113963f6be16ef6ed4f3999725e95bef29681405ac7283.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/cc40432afd0554c723113963f6be16ef6ed4f3999725e95bef29681405ac7283.jpg)

![ccfe506bcadeb36f58f2c6102c90f069afba2c06f9b2c680a953bad8bcf62ce2.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/ccfe506bcadeb36f58f2c6102c90f069afba2c06f9b2c680a953bad8bcf62ce2.jpg)

![cee0f78b01450f06b6e4124ba5887c7961e657836792882684ed3efe198643cb.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/cee0f78b01450f06b6e4124ba5887c7961e657836792882684ed3efe198643cb.jpg)

![cff55bc63e7f9e853a11a25754d9ce1c2ed4ebbecd93b0368486640b27b1decd.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/cff55bc63e7f9e853a11a25754d9ce1c2ed4ebbecd93b0368486640b27b1decd.jpg)

![d1d9f25249eb1478355c13b9fd1ee5187bb1588bb1a8ed5925f3f4cd157995fe.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/d1d9f25249eb1478355c13b9fd1ee5187bb1588bb1a8ed5925f3f4cd157995fe.jpg)

![d3d1179c55fc5e8f232478c5cf4fec6697f1b7cf2f4a454254bf0c18e67fe814.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/d3d1179c55fc5e8f232478c5cf4fec6697f1b7cf2f4a454254bf0c18e67fe814.jpg)

![d4e9e97e63d03912fcf7c9924d5bb381091a98fddd4d812fa9e4b371b4366b7c.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/d4e9e97e63d03912fcf7c9924d5bb381091a98fddd4d812fa9e4b371b4366b7c.jpg)

![d59063c74e09a2a2431a04b28a125ba70003ad3e74863fa2d91bf140fd3daa81.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/d59063c74e09a2a2431a04b28a125ba70003ad3e74863fa2d91bf140fd3daa81.jpg)

![dd4819c2f6e91512abf5416c83b6013a592ede63fcf3a3f499a55e3a04fc28bc.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/dd4819c2f6e91512abf5416c83b6013a592ede63fcf3a3f499a55e3a04fc28bc.jpg)

![e035bc222a19c8e42dbd7ba393e5f505c8f2acbe29fa51848219cfc8ede9701a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/e035bc222a19c8e42dbd7ba393e5f505c8f2acbe29fa51848219cfc8ede9701a.jpg)

![e437a39c317a44f4c56fc7c6c080fd3021186f8cd441000e15945a149de8a163.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/e437a39c317a44f4c56fc7c6c080fd3021186f8cd441000e15945a149de8a163.jpg)

![e49f105a393fba665e026efa7ea778c191ff03197720fb5a1e1a65d43c876871.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/e49f105a393fba665e026efa7ea778c191ff03197720fb5a1e1a65d43c876871.jpg)

![e5a36db1751a199fec3c99ad9a75df490bad69ff7a2e3e1694a8d01c93f2b131.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/e5a36db1751a199fec3c99ad9a75df490bad69ff7a2e3e1694a8d01c93f2b131.jpg)

![e8a69c8a14a11f817ce9cec8f8eabbaa600bb1b951715604f7e58b0937468e94.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/e8a69c8a14a11f817ce9cec8f8eabbaa600bb1b951715604f7e58b0937468e94.jpg)

![e978c273d89af22143ab7f943bd60068fa4e4fc3f6f45a8b6dcb40518a680908.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/e978c273d89af22143ab7f943bd60068fa4e4fc3f6f45a8b6dcb40518a680908.jpg)

![eb777155543673e77ea119454d4002e44290c1e39c75980b0d62960af5a59e32.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/eb777155543673e77ea119454d4002e44290c1e39c75980b0d62960af5a59e32.jpg)

![f1535286e1fd1175efe6617ea4b920af3a10a88e7b51007f3473287d770fb03a.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/f1535286e1fd1175efe6617ea4b920af3a10a88e7b51007f3473287d770fb03a.jpg)

![f37030725f64525a515ed9653b40e3bee691d1b217a8ec04352ed4e2513ac597.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/f37030725f64525a515ed9653b40e3bee691d1b217a8ec04352ed4e2513ac597.jpg)

![f566a3359887745fd07fda0226c9d9408f74761d65a3b4ab174db6217cc74d7f.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/f566a3359887745fd07fda0226c9d9408f74761d65a3b4ab174db6217cc74d7f.jpg)

![f6eb733019a429925b1d0e47b81f4917b905fca6cfb143b705010c7e3dbeb226.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/f6eb733019a429925b1d0e47b81f4917b905fca6cfb143b705010c7e3dbeb226.jpg)

![f739732db4d62843045d015b463a1ed07b8ef940c779c5555868779a73235e7b.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/f739732db4d62843045d015b463a1ed07b8ef940c779c5555868779a73235e7b.jpg)

![fc5a378790a148df52693bea961c5710982551dcca4450528807bef6adb440ea.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/fc5a378790a148df52693bea961c5710982551dcca4450528807bef6adb440ea.jpg)

![fcac5a67d093142b6f24b5a9e2e3d45ee99ec95e6e7248333b3d338ed96e1b21.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/fcac5a67d093142b6f24b5a9e2e3d45ee99ec95e6e7248333b3d338ed96e1b21.jpg)

![fe786c764c46587ac69fae91cc90f381f8d5db4a67d9bcdd9fe8489073aaf51c.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/fe786c764c46587ac69fae91cc90f381f8d5db4a67d9bcdd9fe8489073aaf51c.jpg)

![fec61258d7bfbec3eaf129aa86a9b488b7c9cd9495cb047aa13995b1ebb14e75.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/fec61258d7bfbec3eaf129aa86a9b488b7c9cd9495cb047aa13995b1ebb14e75.jpg)

![ff65734ebc9d85207dbff645ca9935da243e5cb30cf1d0a97cf42ce1a1a565ae.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/images/ff65734ebc9d85207dbff645ca9935da243e5cb30cf1d0a97cf42ce1a1a565ae.jpg)

### Tables

![7f866a90baa0a47ff33d2ca746bd7a2b5c7105f31d2118d33a3a127d90734051.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/tables/7f866a90baa0a47ff33d2ca746bd7a2b5c7105f31d2118d33a3a127d90734051.jpg)

![c07a31e27ba824fc9ede6026939d90278c71269a7d37e7137ca7d1c18ce38a40.jpg](../icml_results/719_Quantifying%20LLM%20Attention-Head%20Stability_%20Implications%20for%20Circuit%20Universality/tables/c07a31e27ba824fc9ede6026939d90278c71269a7d37e7137ca7d1c18ce38a40.jpg)

<a id="paper-0710"></a>

## Modeling Temporal scRNA-seq Data with Latent Gaussian Process and Optimal Transport


### Images

![00f1040298c319316bcd95e6bd9d2e76499a836af7218ed621aefe2fb9ccbdb2.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/00f1040298c319316bcd95e6bd9d2e76499a836af7218ed621aefe2fb9ccbdb2.jpg)

![04e5ab64ead91039e8c05216301647ef67efd184f6ff9ee6fb0c883ed29095ce.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/04e5ab64ead91039e8c05216301647ef67efd184f6ff9ee6fb0c883ed29095ce.jpg)

![1929983239b7a59197b9d01a4b666d642b68257e65918a9befb67199371c63e2.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/1929983239b7a59197b9d01a4b666d642b68257e65918a9befb67199371c63e2.jpg)

![23095e4c51eea5a024a0ee0a1d1cde44243879c9d2da2976da074730e7da8219.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/23095e4c51eea5a024a0ee0a1d1cde44243879c9d2da2976da074730e7da8219.jpg)

![48850758e90e3747fc1c74a48b043558a432227336ee790a26c0d48cadb8e59e.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/48850758e90e3747fc1c74a48b043558a432227336ee790a26c0d48cadb8e59e.jpg)

![6d6357b69bbb693d78fc6efad1eac6e9d611fa55a45239bfc2900de8ca8e19f4.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/6d6357b69bbb693d78fc6efad1eac6e9d611fa55a45239bfc2900de8ca8e19f4.jpg)

![7b601526797dea6db6346cb90a364b397348302a19f3abb2c3664f516ac7ce51.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/7b601526797dea6db6346cb90a364b397348302a19f3abb2c3664f516ac7ce51.jpg)

![7f800e0723fc2b91facc28881215d8ec1aab6758f8e7f2cf5117dfaf4f09b48b.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/7f800e0723fc2b91facc28881215d8ec1aab6758f8e7f2cf5117dfaf4f09b48b.jpg)

![8248bad36f2d542270ffb6512ae1362da6576c940db19f377d93e87b70b2833c.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/8248bad36f2d542270ffb6512ae1362da6576c940db19f377d93e87b70b2833c.jpg)

![8bc0a976ac2a316da092e0e94e8822cf6887c88102af934a321458c3835c64a6.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/8bc0a976ac2a316da092e0e94e8822cf6887c88102af934a321458c3835c64a6.jpg)

![9b9d4021a3f2c022720e3b1412546fb6afd73242fa3492748be194862bd588ae.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/9b9d4021a3f2c022720e3b1412546fb6afd73242fa3492748be194862bd588ae.jpg)

![9baa78d4849640d8e7799f105d98715216777cd6234b5fb8ba52d784e38879b6.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/9baa78d4849640d8e7799f105d98715216777cd6234b5fb8ba52d784e38879b6.jpg)

![b1f147db952490d3f3c872851f0ef40d626580320377e60ed8acd72382287d76.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/b1f147db952490d3f3c872851f0ef40d626580320377e60ed8acd72382287d76.jpg)

![b3b656ba39231949973bb96eb41fa7f391a52fb7435d34fc77c3768dc93fedde.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/b3b656ba39231949973bb96eb41fa7f391a52fb7435d34fc77c3768dc93fedde.jpg)

![bdd48dd538605e41c45264e69e4b86fdc5e3c6a691d059a899c159f5d37e5662.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/bdd48dd538605e41c45264e69e4b86fdc5e3c6a691d059a899c159f5d37e5662.jpg)

![c17b1095281a31d4d05320a81e2f550c812f62d0936eaaa25b2c79928c75d887.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/c17b1095281a31d4d05320a81e2f550c812f62d0936eaaa25b2c79928c75d887.jpg)

![de68f62e8b4f9661882970d5a522a03d2c1fe8c886fec46decd0c2a08e6a0e3e.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/de68f62e8b4f9661882970d5a522a03d2c1fe8c886fec46decd0c2a08e6a0e3e.jpg)

![e2abd0c21419bd2496708e09c68a1dab6d91ef6d45b3ab7060db33c6ff43ad98.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/e2abd0c21419bd2496708e09c68a1dab6d91ef6d45b3ab7060db33c6ff43ad98.jpg)

![e5241e28ec12c730fc4e015b6b6d5700ab86827ec970f36183231f4acfae8bbb.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/e5241e28ec12c730fc4e015b6b6d5700ab86827ec970f36183231f4acfae8bbb.jpg)

![ef5e1c9e0a495d0fd5e989f2c05e6e4e274294f59308c8ac75f2c830c2aac394.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/images/ef5e1c9e0a495d0fd5e989f2c05e6e4e274294f59308c8ac75f2c830c2aac394.jpg)

### Tables

![07ba179bcf08f4eb65975d24085c23e9c32e55cbc31555ba32a6d3e2f834a2da.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/07ba179bcf08f4eb65975d24085c23e9c32e55cbc31555ba32a6d3e2f834a2da.jpg)

![424bede6d4e934ae15bd3022e5e0384b0f3f28225dacc9bfa276d30ce41d630e.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/424bede6d4e934ae15bd3022e5e0384b0f3f28225dacc9bfa276d30ce41d630e.jpg)

![7937e6e978b892846bf0ab0f424e98a72973e6dea193213fe395fd20ab0615f7.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/7937e6e978b892846bf0ab0f424e98a72973e6dea193213fe395fd20ab0615f7.jpg)

![803aec4c09f3b503e487164ff75f5e35da51113ef73e355944adc73f676f8a7b.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/803aec4c09f3b503e487164ff75f5e35da51113ef73e355944adc73f676f8a7b.jpg)

![ab40e23d01ce146efc955c3b0a196965ed99ea04fc950b7a4f2ca8bf7c2a8ac2.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/ab40e23d01ce146efc955c3b0a196965ed99ea04fc950b7a4f2ca8bf7c2a8ac2.jpg)

![b77e73ec9d1d354805e0da84be8452b3cfa7953dd0ec293d01c06859132137c2.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/b77e73ec9d1d354805e0da84be8452b3cfa7953dd0ec293d01c06859132137c2.jpg)

![d53e0d8f41457c72f09529e6a88f312cded8bf57691e35ddc2cb8b8d12b8b93a.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/d53e0d8f41457c72f09529e6a88f312cded8bf57691e35ddc2cb8b8d12b8b93a.jpg)

![eae79334f0ff9bb2224022e1d23eb1a3ca5694c3492e2d6e5e7b52f6c2dd218f.jpg](../icml_results/720_Modeling%20Temporal%20scRNA-seq%20Data%20with%20Latent%20Gaussian%20Process%20and%20Optimal%20Transport/tables/eae79334f0ff9bb2224022e1d23eb1a3ca5694c3492e2d6e5e7b52f6c2dd218f.jpg)

<a id="paper-0711"></a>

## Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making


### Images

![0a41820d83485a45fab8b2cdf4c6712597d73cf322d6163742061f0672de4aac.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/0a41820d83485a45fab8b2cdf4c6712597d73cf322d6163742061f0672de4aac.jpg)

![34d8bcb942017c2e62f2dfcae7dfd58b90eafca3a3c5ff9201def63ea3e0db39.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/34d8bcb942017c2e62f2dfcae7dfd58b90eafca3a3c5ff9201def63ea3e0db39.jpg)

![44f8609973ad0b68865fbffe45bf039a1c54f20bed2307b71bfdedb2c1c472e2.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/44f8609973ad0b68865fbffe45bf039a1c54f20bed2307b71bfdedb2c1c472e2.jpg)

![48a52ddc49de5e34e2bd163f5defaafc67b2b9c5dbd10c25d7ca964c8a313824.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/48a52ddc49de5e34e2bd163f5defaafc67b2b9c5dbd10c25d7ca964c8a313824.jpg)

![66d487facb50790c0c80b8f5501e161388be7e19fb8ac44bdaa5497d80b0adc7.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/66d487facb50790c0c80b8f5501e161388be7e19fb8ac44bdaa5497d80b0adc7.jpg)

![804fb555461280ebf6653c22836673f71d7940ded47b9d30433ea0d13dc0a1c6.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/804fb555461280ebf6653c22836673f71d7940ded47b9d30433ea0d13dc0a1c6.jpg)

![837c015af70015dd123ff2c02da310d4cc7d1bc54002b66a7baafb5fe2b39e9a.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/837c015af70015dd123ff2c02da310d4cc7d1bc54002b66a7baafb5fe2b39e9a.jpg)

![9bc0f79b6ea37b636412f8d546795758b62c01f68b3dc855feeb6cd29e06777e.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/9bc0f79b6ea37b636412f8d546795758b62c01f68b3dc855feeb6cd29e06777e.jpg)

![b8f148b6e9fb700309f8f9815b8b61a89e05870fb7074337063ac2fbc39a6e29.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/b8f148b6e9fb700309f8f9815b8b61a89e05870fb7074337063ac2fbc39a6e29.jpg)

![c373401986c8de07d00481991e77b77ea1ce83f7e28aa22e26a47c2e6d028fc7.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/c373401986c8de07d00481991e77b77ea1ce83f7e28aa22e26a47c2e6d028fc7.jpg)

![d20984e9a4b6696874e6037749bbf1e79e9a2bfceebcb698979d958f1886f270.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/d20984e9a4b6696874e6037749bbf1e79e9a2bfceebcb698979d958f1886f270.jpg)

![e43a56f9e479b923efa1e3d1b416cb7f6096ee94866145c462fcf3680e1c00a4.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/e43a56f9e479b923efa1e3d1b416cb7f6096ee94866145c462fcf3680e1c00a4.jpg)

![f5e995db4f523936da8082af96e6b43b43438e9c86e2b189b3f2a15605a8d951.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/images/f5e995db4f523936da8082af96e6b43b43438e9c86e2b189b3f2a15605a8d951.jpg)

### Tables

![068065f6fb32e1af36d26767b4a5f7eea25cb6aa8ff1e4c36a36e95ac1027a65.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/068065f6fb32e1af36d26767b4a5f7eea25cb6aa8ff1e4c36a36e95ac1027a65.jpg)

![0db44cc992c4f88baf556b3a41f7486ff45a4ec7475ddb9287a01942a0a0649d.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/0db44cc992c4f88baf556b3a41f7486ff45a4ec7475ddb9287a01942a0a0649d.jpg)

![462b9f88892e595960c10861d080ca8182d048eca3f207222fcfff77c6291774.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/462b9f88892e595960c10861d080ca8182d048eca3f207222fcfff77c6291774.jpg)

![4b1e48687dd9f2609bd16de62b756bff499cfe90100e1cc99bd7f6ddad5315c9.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/4b1e48687dd9f2609bd16de62b756bff499cfe90100e1cc99bd7f6ddad5315c9.jpg)

![5ae51a4c6c1b5e3b212fa430378cb91fe8671922878100de4b5babfa2a6291c5.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/5ae51a4c6c1b5e3b212fa430378cb91fe8671922878100de4b5babfa2a6291c5.jpg)

![5d60cc9ba35f81e137c255246cf40fda955e62a0b8067da7476c607f9cf59381.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/5d60cc9ba35f81e137c255246cf40fda955e62a0b8067da7476c607f9cf59381.jpg)

![6d7d50082fc1bf16fec7502b6c44f966e3914d64768720f37e39b356876c2bd9.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/6d7d50082fc1bf16fec7502b6c44f966e3914d64768720f37e39b356876c2bd9.jpg)

![7db0b2fb0b65a9078b1a2dc72a5291b9b86299cacd1abcdb368fed053d403244.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/7db0b2fb0b65a9078b1a2dc72a5291b9b86299cacd1abcdb368fed053d403244.jpg)

![7ddc6974df632296d0f4caa67303f86ebc436751e6ff239c9e94eacd5672f8d5.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/7ddc6974df632296d0f4caa67303f86ebc436751e6ff239c9e94eacd5672f8d5.jpg)

![9be6c7364057c1a7e36dacf824e07ed47d36ac303a965e9dd5565572ada98c84.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/9be6c7364057c1a7e36dacf824e07ed47d36ac303a965e9dd5565572ada98c84.jpg)

![ca79dca7b60df8c1967b7f74ee5071b26b30ee98c5a8fcfdcf593b40819d84d4.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/ca79dca7b60df8c1967b7f74ee5071b26b30ee98c5a8fcfdcf593b40819d84d4.jpg)

![cbad80129f468e1ec979be1cc7acbb0ffaecae59cc1e02dc81cab82ccf703e4a.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/cbad80129f468e1ec979be1cc7acbb0ffaecae59cc1e02dc81cab82ccf703e4a.jpg)

![d4bf54cb38e3dcc6719a62762850bd6d6b02e6bed522d4269b4dfd041f2ad6ad.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/d4bf54cb38e3dcc6719a62762850bd6d6b02e6bed522d4269b4dfd041f2ad6ad.jpg)

![e7a262ff6d39f5ea43986535ed324dd2faf4dbc712de1e235e4b7f47d3221bea.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/e7a262ff6d39f5ea43986535ed324dd2faf4dbc712de1e235e4b7f47d3221bea.jpg)

![e80f6b33fff2d0fa4d1eda6f7c257f71a1c7a1b4ad49a76f85e150a20be55f4e.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/e80f6b33fff2d0fa4d1eda6f7c257f71a1c7a1b4ad49a76f85e150a20be55f4e.jpg)

![ec095a0d7c8d3a98be6a01d34bbaf5bb41e01a00d943ea25d99dba11542a6820.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/ec095a0d7c8d3a98be6a01d34bbaf5bb41e01a00d943ea25d99dba11542a6820.jpg)

![ef6858b667326af756b12cc367e03c3b1d2d0dddcb2bb3ff1b97503ce82322b2.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/ef6858b667326af756b12cc367e03c3b1d2d0dddcb2bb3ff1b97503ce82322b2.jpg)

![ff2e39f8809f436e9d3902e74c348a75bc6e6cb4bfdfb69a6be3a64507c4201c.jpg](../icml_results/721_Model-Based%20Diffusion%20Sampling%20for%20Predictive%20Control%20in%20Offline%20Decision%20Making/tables/ff2e39f8809f436e9d3902e74c348a75bc6e6cb4bfdfb69a6be3a64507c4201c.jpg)

<a id="paper-0712"></a>

## Reading Between the Tokens: Improving Preference Predictions through Mechanistic Forecasting


### Images

![05dcae38d56cc66406e1b01aeb1c3917edf089a8a264ccf12bb9f7293344c65f.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/05dcae38d56cc66406e1b01aeb1c3917edf089a8a264ccf12bb9f7293344c65f.jpg)

![0ca156dab2f2a210ddf9eb62f56ff31da3a12aea6bc938ef22f011e156234d60.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/0ca156dab2f2a210ddf9eb62f56ff31da3a12aea6bc938ef22f011e156234d60.jpg)

![365ee41eb43b72a5b54c6520d4d409cb810145f36fb08b58035f22e748473910.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/365ee41eb43b72a5b54c6520d4d409cb810145f36fb08b58035f22e748473910.jpg)

![488b6171fd028a97f2d09b6c15c2572a1eb6a5b1bbc8e9417524520607482216.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/488b6171fd028a97f2d09b6c15c2572a1eb6a5b1bbc8e9417524520607482216.jpg)

![4bb41c049611a640362b1ebce588d480b1b31c1e8c7095131ddf9ecf914b653e.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/4bb41c049611a640362b1ebce588d480b1b31c1e8c7095131ddf9ecf914b653e.jpg)

![69e02a5159b4681bfa1b5fefc89344804a4a0faff4cf32db141d79ed4bd29072.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/69e02a5159b4681bfa1b5fefc89344804a4a0faff4cf32db141d79ed4bd29072.jpg)

![7172e014cda19c1e39068782dfa58bff5f9f2fa607bf25075a2dab32a6dc7a76.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/7172e014cda19c1e39068782dfa58bff5f9f2fa607bf25075a2dab32a6dc7a76.jpg)

![9ba7da684515b382291a1df5887e84eb76d322c81f96e128cc5db40e36798630.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/9ba7da684515b382291a1df5887e84eb76d322c81f96e128cc5db40e36798630.jpg)

![ba3a6874cbad62e5087fdec191f6980be51c281d15edc272b80671b03f22fcc3.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/ba3a6874cbad62e5087fdec191f6980be51c281d15edc272b80671b03f22fcc3.jpg)

![d2394bf0ed6842adfaab279404f576dafc120accf5936c7affb13f054bd29b91.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/d2394bf0ed6842adfaab279404f576dafc120accf5936c7affb13f054bd29b91.jpg)

![e769321d3da5083fe23438d112b383b799b6231c898fd634067cd68fbcecbf03.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/images/e769321d3da5083fe23438d112b383b799b6231c898fd634067cd68fbcecbf03.jpg)

### Tables

![6aa2b6e729dbbe01e2e8965e5fcf181d289c21a262e14adcbda345b0c9391b7e.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/tables/6aa2b6e729dbbe01e2e8965e5fcf181d289c21a262e14adcbda345b0c9391b7e.jpg)

![70d4145908da089fdf98bba01530a92797a487609acb4d51891c8bf176a75a93.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/tables/70d4145908da089fdf98bba01530a92797a487609acb4d51891c8bf176a75a93.jpg)

![b8bba48d1991ac462672400932586c06c81557c4eaf28a7822479b7b557cd237.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/tables/b8bba48d1991ac462672400932586c06c81557c4eaf28a7822479b7b557cd237.jpg)

![eb7343521a29726fdebfe50fc4e5d2bcae1fff07f3f37f653512ac40a4d3cb6d.jpg](../icml_results/722_Reading%20Between%20the%20Tokens_%20Improving%20Preference%20Predictions%20through%20Mechanistic%20Forecasting/tables/eb7343521a29726fdebfe50fc4e5d2bcae1fff07f3f37f653512ac40a4d3cb6d.jpg)

<a id="paper-0713"></a>

## Don’t Walk the Line: Boundary Guidance for Filtered Generation


### Images

![244eeba1b22b3e544c0610d16da26d3e7380a2cd31b7f32387c16341adbc17db.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/images/244eeba1b22b3e544c0610d16da26d3e7380a2cd31b7f32387c16341adbc17db.jpg)

![bf587f2fd7d17933f2ef68bbbdb0874133ab94a9a7392ad0363c8f1005461606.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/images/bf587f2fd7d17933f2ef68bbbdb0874133ab94a9a7392ad0363c8f1005461606.jpg)

![cb4cbf98ab133fb9b6583f88a30306955b36885d667ad67cddb4eb898056933b.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/images/cb4cbf98ab133fb9b6583f88a30306955b36885d667ad67cddb4eb898056933b.jpg)

### Tables

![01371f1059c862a7e31a0ded2840c04013bc47ea35eadab0fd9eb93b6a102dc6.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/01371f1059c862a7e31a0ded2840c04013bc47ea35eadab0fd9eb93b6a102dc6.jpg)

![1f7ea218e3e3172069baf2147612eb5a0f24bedc3712b87f469738a6fe6a0e6a.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/1f7ea218e3e3172069baf2147612eb5a0f24bedc3712b87f469738a6fe6a0e6a.jpg)

![596123de989b9a5ab8a69de4f9a1e50b12841ce75078cad5505749392f4f43f5.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/596123de989b9a5ab8a69de4f9a1e50b12841ce75078cad5505749392f4f43f5.jpg)

![5c101209bfce4059f8a2e04fe7e13e99c00c70f1c8fd5d5328171dce5dbb07b4.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/5c101209bfce4059f8a2e04fe7e13e99c00c70f1c8fd5d5328171dce5dbb07b4.jpg)

![6db69d811dee5a836cc7a5e78d9327b48c4723cbd5ef072a743c07c855e0106c.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/6db69d811dee5a836cc7a5e78d9327b48c4723cbd5ef072a743c07c855e0106c.jpg)

![6faa0a1be5a7b99a1acff6c98d9965988d7621d87680ddcbeb9a90f264427f7a.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/6faa0a1be5a7b99a1acff6c98d9965988d7621d87680ddcbeb9a90f264427f7a.jpg)

![7c72b9a39c908e1cdabc4d3660375879268e4ba9eb19b532d5a5525b31ec8fcb.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/7c72b9a39c908e1cdabc4d3660375879268e4ba9eb19b532d5a5525b31ec8fcb.jpg)

![9143496afca0e2d0ae80e6ddb82aafc851276b366823ec0f1c8a3070de0def78.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/9143496afca0e2d0ae80e6ddb82aafc851276b366823ec0f1c8a3070de0def78.jpg)

![95cf2fd9eb9ff98bb4ffc9974fc070743fbbd60bd51582454b50871b6d795a7e.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/95cf2fd9eb9ff98bb4ffc9974fc070743fbbd60bd51582454b50871b6d795a7e.jpg)

![95eaba1753d353a86cc4d6a06689c72502d790b49002da2ffa8a4021d6a448af.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/95eaba1753d353a86cc4d6a06689c72502d790b49002da2ffa8a4021d6a448af.jpg)

![9d56b6b8a89fa733ea8faf968ac744dcab6510e714be07b21be68c9923c7fe7a.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/9d56b6b8a89fa733ea8faf968ac744dcab6510e714be07b21be68c9923c7fe7a.jpg)

![a087de24ebb25d28d28b5cf16cd6c24bd197572fa5b4cc949fefd58d42b7f329.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/a087de24ebb25d28d28b5cf16cd6c24bd197572fa5b4cc949fefd58d42b7f329.jpg)

![a348c316c222ebaa8ab6c3b31e142ddc4a080e58f87dd34e663abb8fcc92ca41.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/a348c316c222ebaa8ab6c3b31e142ddc4a080e58f87dd34e663abb8fcc92ca41.jpg)

![ae2951f35edc5d92eae4dddbb99aa137c1183b60bf1d3969d071ce973e618648.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/ae2951f35edc5d92eae4dddbb99aa137c1183b60bf1d3969d071ce973e618648.jpg)

![b04212dba2e6c4f6248515e169987190c79ac2ed4de45cb29be50c34b5fa870f.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/b04212dba2e6c4f6248515e169987190c79ac2ed4de45cb29be50c34b5fa870f.jpg)

![c5583d8258bb5aa47a5094469d803a70251f08e99857ec0465eb395319ab17b3.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/c5583d8258bb5aa47a5094469d803a70251f08e99857ec0465eb395319ab17b3.jpg)

![cb4676f80b5fc567950e54085449c4d5d46ea74ff4dfbd49a2b2d4d157dc5710.jpg](../icml_results/723_Don%E2%80%99t%20Walk%20the%20Line_%20Boundary%20Guidance%20for%20Filtered%20Generation/tables/cb4676f80b5fc567950e54085449c4d5d46ea74ff4dfbd49a2b2d4d157dc5710.jpg)

<a id="paper-0714"></a>

## A Machine-Learned Comorbidity Index


### Images

![145feb30491678a3fbf86d40d9e9b2b7006a65bf985f24bbac43b82b36ba8b1d.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/images/145feb30491678a3fbf86d40d9e9b2b7006a65bf985f24bbac43b82b36ba8b1d.jpg)

![49cf589cd11968ed3a7071dffa27ef4154fa29ac21991311f4d18b0ef925b446.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/images/49cf589cd11968ed3a7071dffa27ef4154fa29ac21991311f4d18b0ef925b446.jpg)

![4f51ab7609b8e865e15573832ae61f18f4f0b1d5195c5669a993f29408720e07.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/images/4f51ab7609b8e865e15573832ae61f18f4f0b1d5195c5669a993f29408720e07.jpg)

![f47cc3e295f071601fb4d563f75af27d433131f1bbf60e3fc89c11c5b8e5d9c1.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/images/f47cc3e295f071601fb4d563f75af27d433131f1bbf60e3fc89c11c5b8e5d9c1.jpg)

![fcc685075a160a85567ff12428e8fc6be16b254fc1a704d3ba42045b36c88148.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/images/fcc685075a160a85567ff12428e8fc6be16b254fc1a704d3ba42045b36c88148.jpg)

### Tables

![3a25d0d26dec52db8acb4d3fb16e3b40badd406697435d5a398eba3e3e2bd03e.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/3a25d0d26dec52db8acb4d3fb16e3b40badd406697435d5a398eba3e3e2bd03e.jpg)

![53217cc0d827a9fcf03cfa3823044048cbc11e23af963e5110a4707074261420.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/53217cc0d827a9fcf03cfa3823044048cbc11e23af963e5110a4707074261420.jpg)

![7261511e3eddf700d4a3d879c2c429ed0700fd7465f189232fab01a5b5ecd698.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/7261511e3eddf700d4a3d879c2c429ed0700fd7465f189232fab01a5b5ecd698.jpg)

![750d2e1baacc774e8489a39869be3507ca91c221688162ecd03fb5c20348be9d.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/750d2e1baacc774e8489a39869be3507ca91c221688162ecd03fb5c20348be9d.jpg)

![7809e7a2c9e6be74a12e48c80d9726dcea950dc78dc5c7a60b168ed3f94a3f1d.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/7809e7a2c9e6be74a12e48c80d9726dcea950dc78dc5c7a60b168ed3f94a3f1d.jpg)

![834b85cd83c52ab4af4e7fd7cec54ca5333a6e92fb8c33f77d38914f0c0fac00.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/834b85cd83c52ab4af4e7fd7cec54ca5333a6e92fb8c33f77d38914f0c0fac00.jpg)

![8aa4b39155cffd2711b9350187ac46bd5b733217812952117c41db9e09804bce.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/8aa4b39155cffd2711b9350187ac46bd5b733217812952117c41db9e09804bce.jpg)

![9d5566e0e4a9ac0bf2257630e95df3fde8b6aac18d5af465eaaaaf4ff065eb88.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/9d5566e0e4a9ac0bf2257630e95df3fde8b6aac18d5af465eaaaaf4ff065eb88.jpg)

![c023b1eccd28397fe6100ab694de20434385295e3c475f9aa7183ee82ec0598c.jpg](../icml_results/724_A%20Machine-Learned%20Comorbidity%20Index/tables/c023b1eccd28397fe6100ab694de20434385295e3c475f9aa7183ee82ec0598c.jpg)

<a id="paper-0715"></a>

## DOT-MoE: Differentiable Optimal Transport for MoEfication


### Images

![a6a9dd7bf6c3a2238e5e825fea752e3af27e7c4137ad15a25f85683525ab1ce5.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/images/a6a9dd7bf6c3a2238e5e825fea752e3af27e7c4137ad15a25f85683525ab1ce5.jpg)

![a9072f56512c22859145d08bc02c621068ab9d12f70ea69b34c0f33793b5e237.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/images/a9072f56512c22859145d08bc02c621068ab9d12f70ea69b34c0f33793b5e237.jpg)

![b4911886681936b8f7bb58c01811ca0daeae592928e2923685450e5f6b3987ff.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/images/b4911886681936b8f7bb58c01811ca0daeae592928e2923685450e5f6b3987ff.jpg)

![d34d7fb115464e9880961251ebf589e08096fb814461b649fd3efa835b38ab2b.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/images/d34d7fb115464e9880961251ebf589e08096fb814461b649fd3efa835b38ab2b.jpg)

### Tables

![29417271cff5dc5ede577e103dc80209b1d48b26d1abde2f4687b84587953938.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/29417271cff5dc5ede577e103dc80209b1d48b26d1abde2f4687b84587953938.jpg)

![358820b231dc6670f7860053d92f949c0f5b5d148224bc9c83fa0d4805c42edc.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/358820b231dc6670f7860053d92f949c0f5b5d148224bc9c83fa0d4805c42edc.jpg)

![48649703229a06bc6bc0aafe6e63336e96ad810d39ec4d024954f4f1df0d4c1e.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/48649703229a06bc6bc0aafe6e63336e96ad810d39ec4d024954f4f1df0d4c1e.jpg)

![49cf6beec3439c932a6f16835a47e712558e73c7d93cb506bd45c983bc5ddc6a.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/49cf6beec3439c932a6f16835a47e712558e73c7d93cb506bd45c983bc5ddc6a.jpg)

![509ec3a779137cc4f021b5d33a64581ce3ce5f0df3ba80a1e91faa2a878d9e5f.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/509ec3a779137cc4f021b5d33a64581ce3ce5f0df3ba80a1e91faa2a878d9e5f.jpg)

![5eba08dad3c33153fdf488787553ed20cd39168acccb69f3695c0dec727ea0c4.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/5eba08dad3c33153fdf488787553ed20cd39168acccb69f3695c0dec727ea0c4.jpg)

![719421450ca660d05bf79b38d34b166abab703a6e659e45d686456a4add92d8b.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/719421450ca660d05bf79b38d34b166abab703a6e659e45d686456a4add92d8b.jpg)

![74cbc31b63b231ba33e7e2eee522c54374fed4b0ec54df09e799438d983d4700.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/74cbc31b63b231ba33e7e2eee522c54374fed4b0ec54df09e799438d983d4700.jpg)

![a95262f4c5ccc9be7b6bc56efde6e1452afc7753358abfb6cabf6c1a767e6dd5.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/a95262f4c5ccc9be7b6bc56efde6e1452afc7753358abfb6cabf6c1a767e6dd5.jpg)

![b4deb63a8a0258615e724b334360ff58baaea242f11fdc561d4a9cfe4f45565b.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/b4deb63a8a0258615e724b334360ff58baaea242f11fdc561d4a9cfe4f45565b.jpg)

![d1a876dae74521d8fb2d89579caee9c0e93b012a3c7560f5b3e4e164d026fba6.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/d1a876dae74521d8fb2d89579caee9c0e93b012a3c7560f5b3e4e164d026fba6.jpg)

![d21a294410895a129d0b677aaf4f0d18e3ef879c9c0b0e12f23ff015ca51985c.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/d21a294410895a129d0b677aaf4f0d18e3ef879c9c0b0e12f23ff015ca51985c.jpg)

![ebc526ba5e49f3022785a8f4350d57f22332220c75e449645e501157eeb4c182.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/ebc526ba5e49f3022785a8f4350d57f22332220c75e449645e501157eeb4c182.jpg)

![f762089a4c268c66a74e7bc0790714b979e111ea00e1452f640bc2db13d5df72.jpg](../icml_results/725_DOT-MoE_%20Differentiable%20Optimal%20Transport%20for%20MoEfication/tables/f762089a4c268c66a74e7bc0790714b979e111ea00e1452f640bc2db13d5df72.jpg)

<a id="paper-0716"></a>

## XRPO: Pushing the Limits of GRPO with Targeted Exploration and Exploitation


### Images

![03b39d07732643bd15e30ed8e33453737ec3ba5a401a30a7174133446eb9ff05.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/images/03b39d07732643bd15e30ed8e33453737ec3ba5a401a30a7174133446eb9ff05.jpg)

![19de24149322bdaf9e6146b8e65707681d041cdbf5cba3ec762c9da35e45b1fc.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/images/19de24149322bdaf9e6146b8e65707681d041cdbf5cba3ec762c9da35e45b1fc.jpg)

![2d20416a8e6c8da695135d1074d64f43541705636dd632f9def53c44a95285d5.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/images/2d20416a8e6c8da695135d1074d64f43541705636dd632f9def53c44a95285d5.jpg)

![603987b9c53db7eddbe09b2a33a184f4d2b46d09bfd2b0539ab3a063bdecc367.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/images/603987b9c53db7eddbe09b2a33a184f4d2b46d09bfd2b0539ab3a063bdecc367.jpg)

![73a95a1e63d6ae4f1e55f601fd6140693c90c2fe0bc76fd1945e3c7b8434826f.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/images/73a95a1e63d6ae4f1e55f601fd6140693c90c2fe0bc76fd1945e3c7b8434826f.jpg)

![ef655ee78b1ae2e92d2da963688a1fb469f1d62bf887a92fd35e7d04a78cd3a1.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/images/ef655ee78b1ae2e92d2da963688a1fb469f1d62bf887a92fd35e7d04a78cd3a1.jpg)

### Tables

![042b747c6935aff2959432705920eb269907a8e41751302f8463c16d7963d6f5.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/042b747c6935aff2959432705920eb269907a8e41751302f8463c16d7963d6f5.jpg)

![1ec463f81a4e23f75fa5ee7a848ab32c67a52be0129c95757d4adec8adcc80f8.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/1ec463f81a4e23f75fa5ee7a848ab32c67a52be0129c95757d4adec8adcc80f8.jpg)

![3f93d7c5cb16f32b8266139c2285daa2fafa19c8754f5072593d52fd607036f8.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/3f93d7c5cb16f32b8266139c2285daa2fafa19c8754f5072593d52fd607036f8.jpg)

![4e94cb3993ee4e7d7064cad0cb6dbd8233cf62a7f6527d1429b46747ea80c936.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/4e94cb3993ee4e7d7064cad0cb6dbd8233cf62a7f6527d1429b46747ea80c936.jpg)

![55cd99dd0368b8297984beb82a6054b0c653a352ba1bea5d2aafedf431099d8f.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/55cd99dd0368b8297984beb82a6054b0c653a352ba1bea5d2aafedf431099d8f.jpg)

![61f1de59010cafbb4f4d077868d1cf0c1e9d9aa7a621cf5fee1d385693c28827.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/61f1de59010cafbb4f4d077868d1cf0c1e9d9aa7a621cf5fee1d385693c28827.jpg)

![65d1b4d13faf8ab9b97fc2d853e6a4276175790c11fe04b58decf0bee716ba6c.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/65d1b4d13faf8ab9b97fc2d853e6a4276175790c11fe04b58decf0bee716ba6c.jpg)

![6758aa46c2cb99af2ea50ee770a9c4a6bdd1140a6d713b0f39134dbd1c6bddfd.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/6758aa46c2cb99af2ea50ee770a9c4a6bdd1140a6d713b0f39134dbd1c6bddfd.jpg)

![769eae65b0e3339d37a4567cfbc4d744d95094458e32be5ec806da97efa5fbd6.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/769eae65b0e3339d37a4567cfbc4d744d95094458e32be5ec806da97efa5fbd6.jpg)

![a64763081fe3a5d1d601468b2832f65c2a7763426ba47cbd7582d2d8eee3c8f4.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/a64763081fe3a5d1d601468b2832f65c2a7763426ba47cbd7582d2d8eee3c8f4.jpg)

![a9e51f47d197774b5d8f723f80fa044573701b41dba77a081a2c0f3da5799af4.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/a9e51f47d197774b5d8f723f80fa044573701b41dba77a081a2c0f3da5799af4.jpg)

![b340621e5a0a6a5846a96ca373c54c8001ae8fde8161005eafc0697ef35fe134.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/b340621e5a0a6a5846a96ca373c54c8001ae8fde8161005eafc0697ef35fe134.jpg)

![d97d6ba4357d3dced023a20a27e4fa13eadcc348cb51cf725cbffafe7a1a61ff.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/d97d6ba4357d3dced023a20a27e4fa13eadcc348cb51cf725cbffafe7a1a61ff.jpg)

![f4bfefb495821a7e59f1a8c0982b721ecc4fb97e14a7987e544e64ecf2cf18ec.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/f4bfefb495821a7e59f1a8c0982b721ecc4fb97e14a7987e544e64ecf2cf18ec.jpg)

![f55440690e6c88779fd4f903232e98882af060e759e6c003e2216ab7c02cf841.jpg](../icml_results/726_XRPO_%20Pushing%20the%20Limits%20of%20GRPO%20with%20Targeted%20Exploration%20and%20Exploitation/tables/f55440690e6c88779fd4f903232e98882af060e759e6c003e2216ab7c02cf841.jpg)

<a id="paper-0717"></a>

## EPSVec: Efficient and Private Synthetic Data Generation via Dataset Vectors


### Images

![05fdc882a9d821f96d8caa3dab20518e61c9275b582b19731308487a3ca1a836.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/05fdc882a9d821f96d8caa3dab20518e61c9275b582b19731308487a3ca1a836.jpg)

![1bc6d3313e9ddb430b99cde2a2b7786e464f0af3273ecb6ef968a7bedf6377eb.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/1bc6d3313e9ddb430b99cde2a2b7786e464f0af3273ecb6ef968a7bedf6377eb.jpg)

![1fd5b516c9b0955d63db2787b39cf4e46dce5877ea91e486defc4b6de2c91710.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/1fd5b516c9b0955d63db2787b39cf4e46dce5877ea91e486defc4b6de2c91710.jpg)

![2f620cd92057acdcb5b584f56787c52029e9c48582c8d80bf2a4074ee458dc49.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/2f620cd92057acdcb5b584f56787c52029e9c48582c8d80bf2a4074ee458dc49.jpg)

![54a99e7b98e98b81f1800878eab432650e6091a0807e7e738cdbd3bf7e3c6fa9.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/54a99e7b98e98b81f1800878eab432650e6091a0807e7e738cdbd3bf7e3c6fa9.jpg)

![5f171fce01d2ef5b4224c8adf0b46c8d48b4abcbf0c936680b8716a1fd1e16b9.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/5f171fce01d2ef5b4224c8adf0b46c8d48b4abcbf0c936680b8716a1fd1e16b9.jpg)

![9837a32dc0a0615c752a2dfb2949e3a347c836d29c64b64efc3e70b0786d6a5b.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/9837a32dc0a0615c752a2dfb2949e3a347c836d29c64b64efc3e70b0786d6a5b.jpg)

![f0c4ffb80a5193b87a827925e5117b30486b4494d364f3f8e0a37596a8a28e9b.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/f0c4ffb80a5193b87a827925e5117b30486b4494d364f3f8e0a37596a8a28e9b.jpg)

![f8cb21a95c7e50a5221e9d2c7c030cf4b038208f97f081096b365d6955170919.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/images/f8cb21a95c7e50a5221e9d2c7c030cf4b038208f97f081096b365d6955170919.jpg)

### Tables

![16161318791110dcbe5864db51d330342608e7b5b61dff3fcd53abf83d8fdbc2.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/16161318791110dcbe5864db51d330342608e7b5b61dff3fcd53abf83d8fdbc2.jpg)

![2d88b0e22a2d4545e76f67463600a4313f5e963a003889e2746f12ccf07f83d6.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/2d88b0e22a2d4545e76f67463600a4313f5e963a003889e2746f12ccf07f83d6.jpg)

![6c756754fd02a4749acfed11699187e5d4bfd52134da079ca96a53263a6cf2f1.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/6c756754fd02a4749acfed11699187e5d4bfd52134da079ca96a53263a6cf2f1.jpg)

![87c1969a7f6ad6363888d4df5894f3f88681ffd4caa08e4e2ab5c58b14a7d185.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/87c1969a7f6ad6363888d4df5894f3f88681ffd4caa08e4e2ab5c58b14a7d185.jpg)

![9f2f67cbf62b2c315b8c0a8c650ef2c68da21e96478ee9e3822786c22ffafd27.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/9f2f67cbf62b2c315b8c0a8c650ef2c68da21e96478ee9e3822786c22ffafd27.jpg)

![b08de7d41536b2280186a7ba4ac9ca07a5d8bd44b365ec7d5b0eaa8b933c1496.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/b08de7d41536b2280186a7ba4ac9ca07a5d8bd44b365ec7d5b0eaa8b933c1496.jpg)

![c3390a039650f1c2d6b1d90b15d3071ed00cdc560a593357f062f40244913de5.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/c3390a039650f1c2d6b1d90b15d3071ed00cdc560a593357f062f40244913de5.jpg)

![d49fef12de6ac5bc09062753203dd186444de2d46285bb6d3a2c17c1f199c333.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/d49fef12de6ac5bc09062753203dd186444de2d46285bb6d3a2c17c1f199c333.jpg)

![f0bcae3649e414fd111b85f99372f6ddfdcb627c11bf0170627588d24842ed19.jpg](../icml_results/727_EPSVec_%20Efficient%20and%20Private%20Synthetic%20Data%20Generation%20via%20Dataset%20Vectors/tables/f0bcae3649e414fd111b85f99372f6ddfdcb627c11bf0170627588d24842ed19.jpg)

<a id="paper-0718"></a>

## RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation


### Images

![0e0755216fecd8642b4a0314ad9802d56324875f0fce41932d8e05efd183df9e.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/0e0755216fecd8642b4a0314ad9802d56324875f0fce41932d8e05efd183df9e.jpg)

![40ce803b270ca31f71a9735c40d05bbcc3c22ecb093cb155364e98e03581ede3.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/40ce803b270ca31f71a9735c40d05bbcc3c22ecb093cb155364e98e03581ede3.jpg)

![48ec5ec41b2d9d1e0b08b05850292a8c2401963748cf88af0dfcc754b0f3805c.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/48ec5ec41b2d9d1e0b08b05850292a8c2401963748cf88af0dfcc754b0f3805c.jpg)

![63d05195d91eb6d1e1615d3c441fb8aec61662902fd4df7f9dab0a747ddd8a1f.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/63d05195d91eb6d1e1615d3c441fb8aec61662902fd4df7f9dab0a747ddd8a1f.jpg)

![78e72db95f588302e9df8e1fa585c8bb1e3017d028165c32dfd3b58417513ac6.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/78e72db95f588302e9df8e1fa585c8bb1e3017d028165c32dfd3b58417513ac6.jpg)

![9e019bfc36583bca5a97a01e3ebb0619ad29a35bcd81dff4706bd24826c03055.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/9e019bfc36583bca5a97a01e3ebb0619ad29a35bcd81dff4706bd24826c03055.jpg)

![a772bccefbcdbe889c8bb8ada37236a31660194af11d5b8400200367c1ac708f.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/a772bccefbcdbe889c8bb8ada37236a31660194af11d5b8400200367c1ac708f.jpg)

![b833ef0d87139f6cd17745671c1d9cf938dd326c6850cb90356eb8d3267277bd.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/b833ef0d87139f6cd17745671c1d9cf938dd326c6850cb90356eb8d3267277bd.jpg)

![e2f5bd611cd90166ed7670b2ef168514f9ba69d2233f70d61b7d86b9b3c127b3.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/e2f5bd611cd90166ed7670b2ef168514f9ba69d2233f70d61b7d86b9b3c127b3.jpg)

![e48ed70a13f5a6882e704f11f6ab25b4ebab85cf16b073d9c80f887bd8f1913d.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/images/e48ed70a13f5a6882e704f11f6ab25b4ebab85cf16b073d9c80f887bd8f1913d.jpg)

### Tables

![74b04781d7c5799e59bb17cca1665e484006be54907ea1f5b86dc9385c9ab310.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/74b04781d7c5799e59bb17cca1665e484006be54907ea1f5b86dc9385c9ab310.jpg)

![9400efd64bf0feb97303af7471b0118790e5e0ade1c46c50743cc84a3f6974d3.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/9400efd64bf0feb97303af7471b0118790e5e0ade1c46c50743cc84a3f6974d3.jpg)

![9a1388747235b86e3081d1ca1890c9f39c50ecb69a533b62ea19b24b0beba797.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/9a1388747235b86e3081d1ca1890c9f39c50ecb69a533b62ea19b24b0beba797.jpg)

![c445eac371b5c3e8e2507784996591b5bd8aad83378f77fc936900a8c05da5c7.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/c445eac371b5c3e8e2507784996591b5bd8aad83378f77fc936900a8c05da5c7.jpg)

![d674f1d04758d91e17dc747c508d42ad498cbeebf1b43bce4979f6f3c75c7c70.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/d674f1d04758d91e17dc747c508d42ad498cbeebf1b43bce4979f6f3c75c7c70.jpg)

![dab973467025605fa2a000ee31651da526b5f352871c86e7c2b44902e2d4a844.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/dab973467025605fa2a000ee31651da526b5f352871c86e7c2b44902e2d4a844.jpg)

![f9aa4320f28f0f828382cdc42e1c7281f4a9eb2d661c433a766f4b8aca3026d5.jpg](../icml_results/728_RSF-GLLM_%20Bridging%20the%20Semantic%20Gap%20in%20Multi-Hop%20Knowledge%20Graph%20QA%20via%20Recurrent%20Soft-Flow%20and%20Deco/tables/f9aa4320f28f0f828382cdc42e1c7281f4a9eb2d661c433a766f4b8aca3026d5.jpg)

<a id="paper-0719"></a>

## Local-Minima-Preserving Polynomial Relaxation of Ising Problems


### Images

![159395ff3f8abc416b6bf0abc80c176f865df192a740a4501919486400d92c3e.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/159395ff3f8abc416b6bf0abc80c176f865df192a740a4501919486400d92c3e.jpg)

![46f4e07c7fc6d9f9a160af4ce5283719fc9c6677311f9d8578fa3ef4498656e8.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/46f4e07c7fc6d9f9a160af4ce5283719fc9c6677311f9d8578fa3ef4498656e8.jpg)

![49825e5cfacd840dd5aeab50790602f5a9a6f876ef921ed4a0280d62d7d4337b.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/49825e5cfacd840dd5aeab50790602f5a9a6f876ef921ed4a0280d62d7d4337b.jpg)

![65714ad3d8e945f9616427fea9f3b320b78aa0eca63f8c2b65fb7b5467248615.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/65714ad3d8e945f9616427fea9f3b320b78aa0eca63f8c2b65fb7b5467248615.jpg)

![aa0ee1515e8a8b5ccfcc083a656c39c3853a8ff3e22ebeeac46d5c63ea622cd7.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/aa0ee1515e8a8b5ccfcc083a656c39c3853a8ff3e22ebeeac46d5c63ea622cd7.jpg)

![cb58104057319dbbfc36cc56ee35af2d8d4878a5261ea64dfe8555068e1a78f6.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/cb58104057319dbbfc36cc56ee35af2d8d4878a5261ea64dfe8555068e1a78f6.jpg)

![f4e09c95ba3e07cf768d62061b7fc7c2f66f570fc6f3603d027b5fa8d4051e95.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/images/f4e09c95ba3e07cf768d62061b7fc7c2f66f570fc6f3603d027b5fa8d4051e95.jpg)

### Tables

![03991ac1980911ea6fae3720cc144d8f62e8b111068e4d8386c91f1daf8ec231.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/tables/03991ac1980911ea6fae3720cc144d8f62e8b111068e4d8386c91f1daf8ec231.jpg)

![91de2cfa60d9a67000aaf6d46b0be7af9242f70a83433a95f62c393c5151463a.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/tables/91de2cfa60d9a67000aaf6d46b0be7af9242f70a83433a95f62c393c5151463a.jpg)

![990f3e9704815cd49eccb0d4c8aad16ca9bd8cdde3a7005a255af3d37423ec00.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/tables/990f3e9704815cd49eccb0d4c8aad16ca9bd8cdde3a7005a255af3d37423ec00.jpg)

![bfa873745a5145f64c4c0721c62634cdd58bbae9cbc836bb2170687fc39ba41c.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/tables/bfa873745a5145f64c4c0721c62634cdd58bbae9cbc836bb2170687fc39ba41c.jpg)

![e62f99fce4bf5178f64366e802c14e30d8d5b1dbfe36e7e77546e692b4f1c949.jpg](../icml_results/729_Local-Minima-Preserving%20Polynomial%20Relaxation%20of%20Ising%20Problems/tables/e62f99fce4bf5178f64366e802c14e30d8d5b1dbfe36e7e77546e692b4f1c949.jpg)

<a id="paper-0720"></a>

## Efficient Continuous-Depth Modeling with GRU Equivalents


### Images

![42cae591e2180ad4c4b5011d4e484182cf118b137e79bf10f140f60978e99918.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/images/42cae591e2180ad4c4b5011d4e484182cf118b137e79bf10f140f60978e99918.jpg)

![873f5e3c5899bf9c96db1de4d86a5570d267d388dd254d1f946f7c1f1d209492.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/images/873f5e3c5899bf9c96db1de4d86a5570d267d388dd254d1f946f7c1f1d209492.jpg)

![b10e77a63818217377ea15691f5de6ceec5f620e2dc3d5663609ea6f7a76b4df.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/images/b10e77a63818217377ea15691f5de6ceec5f620e2dc3d5663609ea6f7a76b4df.jpg)

![c52706e13a89befd19158f69a149c40bc4e9c6ead495cbc5701fa2e557de2892.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/images/c52706e13a89befd19158f69a149c40bc4e9c6ead495cbc5701fa2e557de2892.jpg)

### Tables

![01353af42aa33c46e2c62640ad14123c4e2a947d55f44fd4624bacfefd034418.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/01353af42aa33c46e2c62640ad14123c4e2a947d55f44fd4624bacfefd034418.jpg)

![26410f26a741d7bd3a156d22dbf34e79bbd26284a15af01175afa05cb2d08098.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/26410f26a741d7bd3a156d22dbf34e79bbd26284a15af01175afa05cb2d08098.jpg)

![37c7c7545b73ae13060bb934d6d22156587c6335af6d818659f3d246aa7c23b3.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/37c7c7545b73ae13060bb934d6d22156587c6335af6d818659f3d246aa7c23b3.jpg)

![3e2b4b190fd98a9c87d72fbb61d679b9a47c4ffdb3fe9b9f6fe2cc47b6756274.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/3e2b4b190fd98a9c87d72fbb61d679b9a47c4ffdb3fe9b9f6fe2cc47b6756274.jpg)

![5768ec7c20f5f21fa03ee2b1477284913f0be62566e8c3b488e43693fa1f05a3.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/5768ec7c20f5f21fa03ee2b1477284913f0be62566e8c3b488e43693fa1f05a3.jpg)

![872eeb715ab712cb049c6a1d1e0c576719c9d332bb6440932ada352c1b23e0d3.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/872eeb715ab712cb049c6a1d1e0c576719c9d332bb6440932ada352c1b23e0d3.jpg)

![8961e1fb0d4bec5cec4801f3b9f15ec3031c813a7bbd52dbfdfeebbec528dfe6.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/8961e1fb0d4bec5cec4801f3b9f15ec3031c813a7bbd52dbfdfeebbec528dfe6.jpg)

![9f42ab7076a96e1b0298f166eb95f1d4561b356bce08680fb9a3644eb2cf7b77.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/9f42ab7076a96e1b0298f166eb95f1d4561b356bce08680fb9a3644eb2cf7b77.jpg)

![af97bf3fe67eb797ec4dbbb7c1a6749ecf0f3e638fe311cb378d48191aecc423.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/af97bf3fe67eb797ec4dbbb7c1a6749ecf0f3e638fe311cb378d48191aecc423.jpg)

![afc0537eed05b1caac5ec36b5b2a7431c738d2966b09e3948807ad29ffa1008c.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/afc0537eed05b1caac5ec36b5b2a7431c738d2966b09e3948807ad29ffa1008c.jpg)

![b3f47c1310f49390564eb703755e103dfa2a35567c4e5a112ffab814aa922034.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/b3f47c1310f49390564eb703755e103dfa2a35567c4e5a112ffab814aa922034.jpg)

![b54c475bfc52fef6b4246edf7f1a316a4c6c55bf52f97f39163f0a90d5586c7b.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/b54c475bfc52fef6b4246edf7f1a316a4c6c55bf52f97f39163f0a90d5586c7b.jpg)

![d2d6536f5a2cdfdd4029ea2b6a1c9c04e5e61204bc6e9c2e891054ff6abbb550.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/d2d6536f5a2cdfdd4029ea2b6a1c9c04e5e61204bc6e9c2e891054ff6abbb550.jpg)

![d719033de90e4ac8b8dd9963ecf01ec5b2bece58d240b31bb00e67866954a9f3.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/d719033de90e4ac8b8dd9963ecf01ec5b2bece58d240b31bb00e67866954a9f3.jpg)

![deeae2b77116192c3726f0333deeddd1a59c0172e825aef6a26ea19fa190ef0a.jpg](../icml_results/730_Efficient%20Continuous-Depth%20Modeling%20with%20GRU%20Equivalents/tables/deeae2b77116192c3726f0333deeddd1a59c0172e825aef6a26ea19fa190ef0a.jpg)

<a id="paper-0721"></a>

## Triadic Dynamics Aware Diffusion Posterior Sampling for Inverse Problems: Optimizing Guidance and Stochasticity Schedules


### Images

![131ff5a6dd5d9f9e15c50619d220c8b91755307f44ec0332cdc1cea9a8123b49.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/131ff5a6dd5d9f9e15c50619d220c8b91755307f44ec0332cdc1cea9a8123b49.jpg)

![18207d3ec43ca2787e0aed39a443f283ca28e5718940fa4bff9b482368ff15fe.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/18207d3ec43ca2787e0aed39a443f283ca28e5718940fa4bff9b482368ff15fe.jpg)

![196581984d068d031c1176f200dafe0e02ff8894870fb6f0ed6d8c61fdc3880a.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/196581984d068d031c1176f200dafe0e02ff8894870fb6f0ed6d8c61fdc3880a.jpg)

![1bcdfa536528a86e1bbd45d43cf462145072953627afeb6a8bd8a127dd9c671d.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/1bcdfa536528a86e1bbd45d43cf462145072953627afeb6a8bd8a127dd9c671d.jpg)

![35a9c966b7c1d4212110439cd67efbe908734cfd5a612a38ed34aa35a26fd15a.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/35a9c966b7c1d4212110439cd67efbe908734cfd5a612a38ed34aa35a26fd15a.jpg)

![3800b21746068888babd393730dcfa38262ea81068412cff2ee7c814405b3774.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/3800b21746068888babd393730dcfa38262ea81068412cff2ee7c814405b3774.jpg)

![3cea70d287cb1b76cc89ddf1cb63e17521bc7e67d8cd43c107be6c6a08034f51.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/3cea70d287cb1b76cc89ddf1cb63e17521bc7e67d8cd43c107be6c6a08034f51.jpg)

![44906baef408dd30c3b8299a52f63fb9f2295dcacffd2db6b1cfa9a63dece4a3.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/44906baef408dd30c3b8299a52f63fb9f2295dcacffd2db6b1cfa9a63dece4a3.jpg)

![629f78da212e9886ac1ff9bbac9696c137700e287e3f399aab1fa9c6874be040.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/629f78da212e9886ac1ff9bbac9696c137700e287e3f399aab1fa9c6874be040.jpg)

![65cd04a2ffdac9b89705c6c41f66324f53015638e229ce4c82e1e831cdc4e7f8.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/65cd04a2ffdac9b89705c6c41f66324f53015638e229ce4c82e1e831cdc4e7f8.jpg)

![700bfdf1086546a503a0d75a2ccd109363a7f2aa6653e9d870b956053e4ddeb8.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/700bfdf1086546a503a0d75a2ccd109363a7f2aa6653e9d870b956053e4ddeb8.jpg)

![83d13c0b5117bf5b5b83f0f54b98d399e8d1477cc282d42fa2f8be54e1552862.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/83d13c0b5117bf5b5b83f0f54b98d399e8d1477cc282d42fa2f8be54e1552862.jpg)

![a763bdfa2f42296c93857c680a5053edff7e0b854a3e11b3c656dc2c27e146dd.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/a763bdfa2f42296c93857c680a5053edff7e0b854a3e11b3c656dc2c27e146dd.jpg)

![a80d1ef3497b73de660ab703f1d4a5cbd2ed769ce1f498113a67ef9f275cfd7e.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/a80d1ef3497b73de660ab703f1d4a5cbd2ed769ce1f498113a67ef9f275cfd7e.jpg)

![bbb74c9ae497ed896392870c7a0c4d5871a5c7ebfd60fa83c7d6b3ff70684236.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/bbb74c9ae497ed896392870c7a0c4d5871a5c7ebfd60fa83c7d6b3ff70684236.jpg)

![d64ff41cea05e9e65f948571d7680fc55f150a9b7852b81ed216f24e0b23ae07.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/d64ff41cea05e9e65f948571d7680fc55f150a9b7852b81ed216f24e0b23ae07.jpg)

![dbaa968a3fbe018d05bdf7deb5c77f2b0b6357e85810200192b210d5f594f5f8.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/dbaa968a3fbe018d05bdf7deb5c77f2b0b6357e85810200192b210d5f594f5f8.jpg)

![e6baf3b15190144b6e6f3f35f1cfe359282472a04a30ca0edf9ac637c26f5738.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/e6baf3b15190144b6e6f3f35f1cfe359282472a04a30ca0edf9ac637c26f5738.jpg)

![f20f626025924162ab0cdd0a44caf13f565145c3a8c335cf0bae0c520b1fc3aa.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/f20f626025924162ab0cdd0a44caf13f565145c3a8c335cf0bae0c520b1fc3aa.jpg)

![ffffcf0934dcf6fbe1ae6007285f97a3c6282a4c2f0235c7a83154a38cc68305.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/images/ffffcf0934dcf6fbe1ae6007285f97a3c6282a4c2f0235c7a83154a38cc68305.jpg)

### Tables

![1badd676b5ce0fef0e7a92c747d20d4756fb1fdf68c8e060930fc24aea22a426.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/1badd676b5ce0fef0e7a92c747d20d4756fb1fdf68c8e060930fc24aea22a426.jpg)

![1c1fee9389ef9b14c17d26bdca0f73bbed971d1577a0ae0a4504f88003afed8d.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/1c1fee9389ef9b14c17d26bdca0f73bbed971d1577a0ae0a4504f88003afed8d.jpg)

![238383136317c5c16e50823cef1d772a771a9a47ee1f65241eed6b88ec2b4274.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/238383136317c5c16e50823cef1d772a771a9a47ee1f65241eed6b88ec2b4274.jpg)

![2973e6bbe53af6c256d3f5bffdee363a5f8c6401ab71d50643a3a9d86d05b318.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/2973e6bbe53af6c256d3f5bffdee363a5f8c6401ab71d50643a3a9d86d05b318.jpg)

![35f8b5fdbc58269bd8b6b6883128d39061fca773d11653b5a6d69129c77ae4c8.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/35f8b5fdbc58269bd8b6b6883128d39061fca773d11653b5a6d69129c77ae4c8.jpg)

![4531d05d105736daecf253af17f5915be5dc1874172905e730afd7919cfc577e.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/4531d05d105736daecf253af17f5915be5dc1874172905e730afd7919cfc577e.jpg)

![55e7a5e4e1650c282a2ee220d394900f0a4464a97ffbca73a98b88052e3cb168.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/55e7a5e4e1650c282a2ee220d394900f0a4464a97ffbca73a98b88052e3cb168.jpg)

![7df373973b602f09c0cf2dd8264f3ff45c95448f605a896ffc99ca6185c0bbb3.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/7df373973b602f09c0cf2dd8264f3ff45c95448f605a896ffc99ca6185c0bbb3.jpg)

![845b94e9a436085dc727cdbb3856d695fb578c46b7e464acaaf450b6f0664c0d.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/845b94e9a436085dc727cdbb3856d695fb578c46b7e464acaaf450b6f0664c0d.jpg)

![86ab732ce6c5870caea05c120ef0dcda28d1a6cf52a517b592c1fede2c3cf32b.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/86ab732ce6c5870caea05c120ef0dcda28d1a6cf52a517b592c1fede2c3cf32b.jpg)

![ab260c9222c788b80ee217d299b2fd84c9c02772b5c41980b39aa1064ebab728.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/ab260c9222c788b80ee217d299b2fd84c9c02772b5c41980b39aa1064ebab728.jpg)

![bfe4bef83a9fabd2b265874e7d6f174b573e70a48959952bdc998f3b2f575060.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/bfe4bef83a9fabd2b265874e7d6f174b573e70a48959952bdc998f3b2f575060.jpg)

![df4fd43b995d1e63d258cc40a3ad298848999abf599e9d662512dabc9f9ddeec.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/df4fd43b995d1e63d258cc40a3ad298848999abf599e9d662512dabc9f9ddeec.jpg)

![efc0bd67adf9df17a74abb62eba86384c7b7012af671b879557e1795af75d5f4.jpg](../icml_results/731_Triadic%20Dynamics%20Aware%20Diffusion%20Posterior%20Sampling%20for%20Inverse%20Problems_%20Optimizing%20Guidance%20and%20St/tables/efc0bd67adf9df17a74abb62eba86384c7b7012af671b879557e1795af75d5f4.jpg)

<a id="paper-0722"></a>

## Adversarially Robust Approximate Furthest Neighbor


### Tables

![f4cdb85ed6c589479703a7820117ea17a24b32ea37d26222183a4dcc911abaa5.jpg](../icml_results/733_Adversarially%20Robust%20Approximate%20Furthest%20Neighbor/tables/f4cdb85ed6c589479703a7820117ea17a24b32ea37d26222183a4dcc911abaa5.jpg)

<a id="paper-0723"></a>

## Improved Dynamic Algorithm for Non-monotone Submodular Maximization under Cardinality Constraint


### Tables

![54ca090fe12f28070ae741efac532b2c5cb4db27e66692b173e35735add26261.jpg](../icml_results/734_Improved%20Dynamic%20Algorithm%20for%20Non-monotone%20Submodular%20Maximization%20under%20Cardinality%20Constraint/tables/54ca090fe12f28070ae741efac532b2c5cb4db27e66692b173e35735add26261.jpg)

<a id="paper-0724"></a>

## Conformal C2ST: Turning weak classifiers into strong two-sample tests


### Images

![01d46cb1d354385d6e3458e27c8772dd7eb1b07d13f8de895b033831c38487e4.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/01d46cb1d354385d6e3458e27c8772dd7eb1b07d13f8de895b033831c38487e4.jpg)

![20a77588233f6b1a16976129a644286046c84edaab98dca17eb6ddd79abfb995.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/20a77588233f6b1a16976129a644286046c84edaab98dca17eb6ddd79abfb995.jpg)

![650eb9e99c6058733daf571be79f79df0c258a0784619a6b6c97cf02c05abc59.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/650eb9e99c6058733daf571be79f79df0c258a0784619a6b6c97cf02c05abc59.jpg)

![873cc382899970407677ece07b113638fb5a8f8374366dcd2a59913d6a48a609.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/873cc382899970407677ece07b113638fb5a8f8374366dcd2a59913d6a48a609.jpg)

![98d4a88daa263c7ce69ca100e576196cc288d22d8b008e1cca297397876abbce.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/98d4a88daa263c7ce69ca100e576196cc288d22d8b008e1cca297397876abbce.jpg)

![9b11d0e81331cea64604a9333f77cd1c9644564d597fa6cb52d42f69ad4fac98.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/9b11d0e81331cea64604a9333f77cd1c9644564d597fa6cb52d42f69ad4fac98.jpg)

![b51395aca2766b730338804af2c856eab40c4a41a8f1f7d86a3573f8d99fd527.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/b51395aca2766b730338804af2c856eab40c4a41a8f1f7d86a3573f8d99fd527.jpg)

![b630000d2ea9da7ab9fc8df801ebc6f0099f706418325ba163ab20c2c07e25b0.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/b630000d2ea9da7ab9fc8df801ebc6f0099f706418325ba163ab20c2c07e25b0.jpg)

![d81bac1135dd3a11c5b08bcfc74e90006037bb213ded37b49a77bf5a23379ea7.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/d81bac1135dd3a11c5b08bcfc74e90006037bb213ded37b49a77bf5a23379ea7.jpg)

![ec98a73a1f56fcc1ff6318b24ed8f67274c4c53f22477c64722e4cb24f19ca87.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/ec98a73a1f56fcc1ff6318b24ed8f67274c4c53f22477c64722e4cb24f19ca87.jpg)

![fae31ad3654de5e28479aa794be4f6f9586af1b102dc556bf1361e1b8d4c8f23.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/fae31ad3654de5e28479aa794be4f6f9586af1b102dc556bf1361e1b8d4c8f23.jpg)

![fb414c88ed83df4575b1f6512ff85bcb340d06900f1e8ad21f29b410193e3af7.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/images/fb414c88ed83df4575b1f6512ff85bcb340d06900f1e8ad21f29b410193e3af7.jpg)

### Tables

![018b295cbf0435bc4a1cf7cf8449b26832a42b26f6cb92e64c8a4f1106bcb863.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/tables/018b295cbf0435bc4a1cf7cf8449b26832a42b26f6cb92e64c8a4f1106bcb863.jpg)

![87f5cdafd6c11ac99065439ebe4ae6f3bf68b3fb176615005eeeb6d5d4e9f038.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/tables/87f5cdafd6c11ac99065439ebe4ae6f3bf68b3fb176615005eeeb6d5d4e9f038.jpg)

![cd86b8b29deaa4d4845918d0fb494f3b0435ae739ddc0329a22a554971368c6d.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/tables/cd86b8b29deaa4d4845918d0fb494f3b0435ae739ddc0329a22a554971368c6d.jpg)

![d87f0a396d48359b81f28d3ab2348275feb41fb861bcc1bba976d859418ced3c.jpg](../icml_results/735_Conformal%20C2ST_%20Turning%20weak%20classifiers%20into%20strong%20two-sample%20tests/tables/d87f0a396d48359b81f28d3ab2348275feb41fb861bcc1bba976d859418ced3c.jpg)

<a id="paper-0725"></a>

## Learning Junta Distributions, Quantum Junta States, and QAC$^0$ Circuits


### Tables

![b35def018782f04bf3a7aabef4b8782e9422d702ab696bc5f6ab914eb2dab7ed.jpg](../icml_results/736_Learning%20Junta%20Distributions%2C%20Quantum%20Junta%20States%2C%20and%20QAC%24%5E0%24%20Circuits/tables/b35def018782f04bf3a7aabef4b8782e9422d702ab696bc5f6ab914eb2dab7ed.jpg)

<a id="paper-0726"></a>

## Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions

### Images

![01a89c00cbdefbd13c306f37e966a236b1a9f4189966e3ac8dcc966b91d4cf5e.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/01a89c00cbdefbd13c306f37e966a236b1a9f4189966e3ac8dcc966b91d4cf5e.jpg)

![02600327bbd2a81d5e9e71b47a83e9d35a36a8808f20da6a04b1183330916d94.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/02600327bbd2a81d5e9e71b47a83e9d35a36a8808f20da6a04b1183330916d94.jpg)

![031d3c8bad1fb325805c1367817aa5cb4bfb92c50d1d074524da6edd8b9bd484.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/031d3c8bad1fb325805c1367817aa5cb4bfb92c50d1d074524da6edd8b9bd484.jpg)

![069aa44bf56145da84bdee835026090aadf79d2b37d41eadc80e8178bcd83189.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/069aa44bf56145da84bdee835026090aadf79d2b37d41eadc80e8178bcd83189.jpg)

![0f94d10e3fca218272d073ad9c012d08ef32dcdd0defb577425e542cd4e3cb46.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/0f94d10e3fca218272d073ad9c012d08ef32dcdd0defb577425e542cd4e3cb46.jpg)

![1b180484511e8e51ba490248cdc20ff6e1705a3a798a1bfacff28d86b9af4015.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/1b180484511e8e51ba490248cdc20ff6e1705a3a798a1bfacff28d86b9af4015.jpg)

![1b55e136e42779b10ac2a8c72310c905a9ae21861acae229062475fa1fbd92fe.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/1b55e136e42779b10ac2a8c72310c905a9ae21861acae229062475fa1fbd92fe.jpg)

![1ff3c25eca898dcf4b46f82270ca849943237448187f6ee5026e5e4f722499ea.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/1ff3c25eca898dcf4b46f82270ca849943237448187f6ee5026e5e4f722499ea.jpg)

![2437e1589b8a8d0be12616dc4c30519bf7ced7173ac4d56d9b1d4bb3021ecd09.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/2437e1589b8a8d0be12616dc4c30519bf7ced7173ac4d56d9b1d4bb3021ecd09.jpg)

![24c4bef5df34eacdde4a6fd26ae6f87526371df1a873444033442143f69a4b30.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/24c4bef5df34eacdde4a6fd26ae6f87526371df1a873444033442143f69a4b30.jpg)

![262a4c07860c4934a9e69aee133ac8cd7162a0f937e4816835d037e1c11b74d3.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/262a4c07860c4934a9e69aee133ac8cd7162a0f937e4816835d037e1c11b74d3.jpg)

![3e35b00b9d1d7d1d7343b31efb2e8adb94aa11ac7b9305d647f477f43271fa85.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/3e35b00b9d1d7d1d7343b31efb2e8adb94aa11ac7b9305d647f477f43271fa85.jpg)

![452f5ffe953e214bceec95a4bca97f5666eb3e8bca100d0a81bf8a109b159a62.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/452f5ffe953e214bceec95a4bca97f5666eb3e8bca100d0a81bf8a109b159a62.jpg)

![4976ef0542aa0c569fbc253a6bffb9f228e0e11286186f2ef96f8908875fb72d.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/4976ef0542aa0c569fbc253a6bffb9f228e0e11286186f2ef96f8908875fb72d.jpg)

![4c11bf853834e12f0acc387575d688bae3f66bceacbed2b44fb99031378adf90.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/4c11bf853834e12f0acc387575d688bae3f66bceacbed2b44fb99031378adf90.jpg)

![4dd784182d85f3531b744c9cd3f3be1f0ee0cdd9a499fea131f7c158b1badf26.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/4dd784182d85f3531b744c9cd3f3be1f0ee0cdd9a499fea131f7c158b1badf26.jpg)

![51db52041eef99d1d5b8b8f334346e5683d96b6cf997d6e3169d19f6649b75c7.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/51db52041eef99d1d5b8b8f334346e5683d96b6cf997d6e3169d19f6649b75c7.jpg)

![5224367fe7e3ff9cfd315e1229dca0151b4e05504b46718c61c1ab4fc7bcab67.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/5224367fe7e3ff9cfd315e1229dca0151b4e05504b46718c61c1ab4fc7bcab67.jpg)

![52bd7544367b19dfb2e6a64cd8c99721ec840ba261d45b2e7854ffc633c23739.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/52bd7544367b19dfb2e6a64cd8c99721ec840ba261d45b2e7854ffc633c23739.jpg)

![59ed3f1b214efe7f0463b7f5ce7d0b242e19e071f7da74b0f626dec2d72436ba.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/59ed3f1b214efe7f0463b7f5ce7d0b242e19e071f7da74b0f626dec2d72436ba.jpg)

![5c067cab3d9aab54c9740249f46320a04b157bacf10c5d6cb3765b9dad5c2024.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/5c067cab3d9aab54c9740249f46320a04b157bacf10c5d6cb3765b9dad5c2024.jpg)

![6e82b2c6964030cd33edd4dfaed6e4b49fba88b8ddf79b9ebd6c07ab650b56ad.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/6e82b2c6964030cd33edd4dfaed6e4b49fba88b8ddf79b9ebd6c07ab650b56ad.jpg)

![75aa2527d179c9c4312356bcfb94d24611b07b8dfb90a05b6bd984359a3bc9eb.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/75aa2527d179c9c4312356bcfb94d24611b07b8dfb90a05b6bd984359a3bc9eb.jpg)

![7a978f9213768aa1fcc2504630ded75c9485d5b55bfa883d0f5c2947431e28e6.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/7a978f9213768aa1fcc2504630ded75c9485d5b55bfa883d0f5c2947431e28e6.jpg)

![800210751d43b70a6f79c940d4b3cda5b8cde5cfad89dddfe6b2309074fe1140.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/800210751d43b70a6f79c940d4b3cda5b8cde5cfad89dddfe6b2309074fe1140.jpg)

![9b00587f80c075594259f7d9730e663fbd5a45658de40c655e1b310f4ae39021.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/9b00587f80c075594259f7d9730e663fbd5a45658de40c655e1b310f4ae39021.jpg)

![9b0579eb307d7f40c4f9330e5dd6cc8e87165a3b5896dd5adc393468b6db60eb.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/9b0579eb307d7f40c4f9330e5dd6cc8e87165a3b5896dd5adc393468b6db60eb.jpg)

![9b160492d34f0e2fa883bdc3d9b00daa57ccca73985272e1775424d71acd8a61.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/9b160492d34f0e2fa883bdc3d9b00daa57ccca73985272e1775424d71acd8a61.jpg)

![9e05d1cf73c0d563ff2ce6e149fbfb0e543967a333ebab63c72596afeee7f07f.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/9e05d1cf73c0d563ff2ce6e149fbfb0e543967a333ebab63c72596afeee7f07f.jpg)

![a889f24f95368772d3a0d7616e1ed29b80b264a4f9cfb415537e175bde9f1f62.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/a889f24f95368772d3a0d7616e1ed29b80b264a4f9cfb415537e175bde9f1f62.jpg)

![b4c7c6cf0956bd872b75b09dfe75c46291f1c292b0a879f5302a3ab196a71de3.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/b4c7c6cf0956bd872b75b09dfe75c46291f1c292b0a879f5302a3ab196a71de3.jpg)

![b6bbba23636b69d1bdd30e2d6d8179c880f95605550f3eb898d348377e99939b.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/b6bbba23636b69d1bdd30e2d6d8179c880f95605550f3eb898d348377e99939b.jpg)

![ba5a7eb42ddc67a3473aadf8f387e99b7b798866636ff46e31f2f490ac92c2e6.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/ba5a7eb42ddc67a3473aadf8f387e99b7b798866636ff46e31f2f490ac92c2e6.jpg)

![beaeac8c1f58ba9e39e1499afc7f6f832d1fe9729a6f869510063f7a7c11c788.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/beaeac8c1f58ba9e39e1499afc7f6f832d1fe9729a6f869510063f7a7c11c788.jpg)

![c096b7f3bfcdb57aabc6489b622ffe407dcd6aeb03251bada15cd7240dcd8d96.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/c096b7f3bfcdb57aabc6489b622ffe407dcd6aeb03251bada15cd7240dcd8d96.jpg)

![c6e5c6ee28dc794ba9c5fc3d81f5cb66d70fcd1cf5c4f7728e9e21f665fe4d3d.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/c6e5c6ee28dc794ba9c5fc3d81f5cb66d70fcd1cf5c4f7728e9e21f665fe4d3d.jpg)

![c909ab23ad683d8e058bced853e87cfe09b621637488db17574bf8604fb06c65.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/c909ab23ad683d8e058bced853e87cfe09b621637488db17574bf8604fb06c65.jpg)

![cc329c4cdbd058b4ee9fe8ea19d83303d9b081916b732e447426927f8a68bd21.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/cc329c4cdbd058b4ee9fe8ea19d83303d9b081916b732e447426927f8a68bd21.jpg)

![d92c153c7b1b87fef9d4d49494fbaec9f194cf2a77f9ae4699d1bc8f2fd92dd3.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/d92c153c7b1b87fef9d4d49494fbaec9f194cf2a77f9ae4699d1bc8f2fd92dd3.jpg)

![df0301ac9ba687e002ed7d96839ba0b563dac94343919a7c1e2b25e90bd59a56.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/df0301ac9ba687e002ed7d96839ba0b563dac94343919a7c1e2b25e90bd59a56.jpg)

![df6ca8a5cf02de074c9712102c4889a3859503005294036203f55ef1af10edda.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/df6ca8a5cf02de074c9712102c4889a3859503005294036203f55ef1af10edda.jpg)

![e20b6378bc2f211948a2c353990e4bcb594c741295aa27aac0095018a28b1ee1.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/e20b6378bc2f211948a2c353990e4bcb594c741295aa27aac0095018a28b1ee1.jpg)

![e50c8cbd71871a9c48f3e5c25923541685e3f38a66d5852938263e1b5b7c4d14.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/e50c8cbd71871a9c48f3e5c25923541685e3f38a66d5852938263e1b5b7c4d14.jpg)

![e8dccc6de09a378a16b19ef83e45f981bdbc6e2ad9c9cdbd184c0c1d0fe19771.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/e8dccc6de09a378a16b19ef83e45f981bdbc6e2ad9c9cdbd184c0c1d0fe19771.jpg)

![ea45dd1e8f2815f87896560e390227ed1ab62eed8d4bd26fbc6d196cd947ee18.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/ea45dd1e8f2815f87896560e390227ed1ab62eed8d4bd26fbc6d196cd947ee18.jpg)

![fdff2d72355bf559d742be46bb770c736afa6e25e324eddb6951e07ca431e951.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/fdff2d72355bf559d742be46bb770c736afa6e25e324eddb6951e07ca431e951.jpg)

![fe0c982bb335f25dcbe71b24e9c213d5edaf82332785ebac3a83251de51d7aeb.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/fe0c982bb335f25dcbe71b24e9c213d5edaf82332785ebac3a83251de51d7aeb.jpg)

![ff62da805ea9683f4c8da31d2651e4fe810df8f883ca0a04b541d5969a2a4327.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/images/ff62da805ea9683f4c8da31d2651e4fe810df8f883ca0a04b541d5969a2a4327.jpg)

### Tables

![0325d297f300f3c758931fe9e74f2dccf56cd9b14b3821659078f405e4bf36c1.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/0325d297f300f3c758931fe9e74f2dccf56cd9b14b3821659078f405e4bf36c1.jpg)

![036afae03bc43ea40ed449779766962163f304db171560086d3d9f0639d1defb.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/036afae03bc43ea40ed449779766962163f304db171560086d3d9f0639d1defb.jpg)

![08d7877f4e538b5665b67d23fe4ed154e9b1e51d41c77087f9b3e0b7e428a6c4.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/08d7877f4e538b5665b67d23fe4ed154e9b1e51d41c77087f9b3e0b7e428a6c4.jpg)

![150d294ef49e9922b3768e89a64f87960129aa404ac194f2d12b32627188fed6.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/150d294ef49e9922b3768e89a64f87960129aa404ac194f2d12b32627188fed6.jpg)

![1ad8a2b294e2b744e430eac418f41c25b4d33639187a4d6ff3bb4718e26f0a07.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/1ad8a2b294e2b744e430eac418f41c25b4d33639187a4d6ff3bb4718e26f0a07.jpg)

![23ffa9cb19b4f1614e9ce0f44ac644f81fa34d0b227a2529d3ded5fa8e69d41a.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/23ffa9cb19b4f1614e9ce0f44ac644f81fa34d0b227a2529d3ded5fa8e69d41a.jpg)

![31f0c8b2ed900c7772e6bbfc5bf3af8e4fa64dea264c3481a80856c15885b35c.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/31f0c8b2ed900c7772e6bbfc5bf3af8e4fa64dea264c3481a80856c15885b35c.jpg)

![3cfb0293fc1e23baede6a73d25b6c21f3cdcef6d5db991bb2f94d60918b2b27e.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/3cfb0293fc1e23baede6a73d25b6c21f3cdcef6d5db991bb2f94d60918b2b27e.jpg)

![47eca2a7df37887a1abf2a266cff1850b84e1af85894ebaa046edd972d8d2195.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/47eca2a7df37887a1abf2a266cff1850b84e1af85894ebaa046edd972d8d2195.jpg)

![503487653fa4cf23e13b2de1940b8a9b6fb4513795cde786d9e3b9c0ba0c6e27.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/503487653fa4cf23e13b2de1940b8a9b6fb4513795cde786d9e3b9c0ba0c6e27.jpg)

![58d53c7640b005edf517f554383341635b944d228f5630399ffe945df755cc8b.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/58d53c7640b005edf517f554383341635b944d228f5630399ffe945df755cc8b.jpg)

![6bb71bd7501c6455857f3854db7b86a3647a09f0e3d35898e24e72bc9b2c4fe8.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/6bb71bd7501c6455857f3854db7b86a3647a09f0e3d35898e24e72bc9b2c4fe8.jpg)

![74e55497bffad9ca3cbc337a17807c85ba4ae1c3550034a358d258ff19556c40.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/74e55497bffad9ca3cbc337a17807c85ba4ae1c3550034a358d258ff19556c40.jpg)

![75095a85b9ff16f7d4efa6929294315002c54f932df6226f22ebe45fb11d30ca.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/75095a85b9ff16f7d4efa6929294315002c54f932df6226f22ebe45fb11d30ca.jpg)

![76fa17bbad2787cd6bc9713821c9d1d88673d1aef644ae4a5e2d8c50d372c288.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/76fa17bbad2787cd6bc9713821c9d1d88673d1aef644ae4a5e2d8c50d372c288.jpg)

![77d0419c1aa5a05c409c8db34f8d1b92926c132bfe92d930dd414543cba3977b.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/77d0419c1aa5a05c409c8db34f8d1b92926c132bfe92d930dd414543cba3977b.jpg)

![7a5b4e8c32ebc7bddb7f1d5272e2fe6b98aaa05a0746751ad127905223dd8e2c.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/7a5b4e8c32ebc7bddb7f1d5272e2fe6b98aaa05a0746751ad127905223dd8e2c.jpg)

![8edb5943775973edd5272d5d6dfd434d76c977820446b3c4a1a664411d7608a7.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/8edb5943775973edd5272d5d6dfd434d76c977820446b3c4a1a664411d7608a7.jpg)

![ae0fddd5bde07eef9f9d5fcc277efd645d7bfb735e529057640fb5cff8a45a60.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/ae0fddd5bde07eef9f9d5fcc277efd645d7bfb735e529057640fb5cff8a45a60.jpg)

![ba1665b724498559e816e754ad3ab4b20382bbc900d832f2390dd937fa98baed.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/ba1665b724498559e816e754ad3ab4b20382bbc900d832f2390dd937fa98baed.jpg)

![bda5a713d792488b6647c8ab6ea8491a20db08b2f903dad5644750286fa29a0f.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/bda5a713d792488b6647c8ab6ea8491a20db08b2f903dad5644750286fa29a0f.jpg)

![ce352f57b21311d08ad5ae21a8d379e0090d12ba4d7fa1d712ef5ba143afc015.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/ce352f57b21311d08ad5ae21a8d379e0090d12ba4d7fa1d712ef5ba143afc015.jpg)

![d298cdd0110587a5611e2aa572f41f0bdb58ce5f2c3ffb3ce16eda0c0b8ba37d.jpg](../icml_results/737_Towards%20Steering%20without%20Sacrifice_%20Principled%20Training%20of%20Steering%20Vectors%20for%20Prompt-only%20Interven/tables/d298cdd0110587a5611e2aa572f41f0bdb58ce5f2c3ffb3ce16eda0c0b8ba37d.jpg)
