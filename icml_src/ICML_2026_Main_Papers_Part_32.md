# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 32 of 197

## 目录 (Table of Contents)

1. [Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality](#paper-1024)
2. [Olmix: A Framework for Data Mixing Throughout LM Development](#paper-1025)
3. [How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning](#paper-1026)
4. [Smooth Multi-Policy Causal Effect Estimation in Longitudinal Settings](#paper-1027)
5. [Discovering Symmetry Groups with Flow Matching](#paper-1028)
6. [MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs](#paper-1029)
7. [Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens](#paper-1030)
8. [SimpleGPT: Improving GPT via A Simple Normalization Strategy](#paper-1031)
9. [Multi-Agent Reinforcement Learning with Submodular Reward](#paper-1032)
10. [3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification](#paper-1033)
11. [XSpecMesh: Quality-Preserving Auto-Regressive Mesh Generation Acceleration via Multi-Head Speculative Decoding](#paper-1034)
12. [Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](#paper-1035)
13. [Scam2Prompt: A Scalable Framework for Auditing Malicious Scam Endpoints in Production LLMs](#paper-1036)
14. [AudioChat: Unified Audio Storytelling, Editing, and Understanding with Transfusion Forcing](#paper-1037)
15. [Beyond External Monitors: Enhancing Transparency of Large Language Models for Easier Monitoring](#paper-1038)
16. [Adversarially Robust Control of Conditional Value-at-Risk via Rockafellar-Uryasev Conformal Inference](#paper-1039)
17. [SMILE: Extended Deep Submodular Function-Based Instruction and In-context Learning Demonstration Selection](#paper-1040)
18. [Autoregression with Self-Token Prediction](#paper-1041)
19. [SPR: A Structured Prompt Refinement Network for Modality Missing](#paper-1042)
20. [Beyond Logits: Coherent Hallucination Mitigation via Attention Contrastive Decoding](#paper-1043)
21. [Noise as a Natural Regularizer in Markov Decision Processes: Connecting Environmental Stochasticity and Policy Simplicity](#paper-1044)
22. [Beyond Blind Noising: Disentangled Visual Rectification for Hallucination Mitigation in MLLMs](#paper-1045)
23. [RELO: Reinforcement Learning to Localize for Visual Object Tracking](#paper-1046)
24. [Mixtures of Geodesic Factor Analyzers on Riemannian Homogeneous Spaces](#paper-1047)
25. [Beyond Token-level Supervision: Unlocking the Potential of Decoding-based Regression via Reinforcement Learning](#paper-1048)
26. [OmniVideo-R1: Reinforcing Audio-visual Reasoning with Query Intention and Modality Attention](#paper-1049)
27. [DynaTok: Token-Based 4D Reconstruction from Partial Point Clouds](#paper-1050)
28. [AppWorld-UL: Benchmarking Diverse Agent-User Interactions for Tool-Use](#paper-1051)
29. [A Bayesian Approach to Quantify the Uncertainty of Human Ratings in a Single-Instance Multimodal Framework](#paper-1052)
30. [Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning](#paper-1053)
31. [Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](#paper-1054)
32. [Multiview Self-Representation Learning across Heterogeneous Views](#paper-1055)
33. [Discrete Tilt Matching](#paper-1056)

---


<a id="paper-1024"></a>

## Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality

### Images

![03594c0fa775d8d83a64205f6bd1f7c2eac65fd8c56093bb779d1551cf7b12b0.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/03594c0fa775d8d83a64205f6bd1f7c2eac65fd8c56093bb779d1551cf7b12b0.jpg)

![4bf3515e85a60c6e207d55de53a85871df6ec55752edcb3960da22f0d8138700.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/4bf3515e85a60c6e207d55de53a85871df6ec55752edcb3960da22f0d8138700.jpg)

![54682ca1e4a141b90282fa91c510b5ab05eabab9daa653a96a0df57b4edf044f.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/54682ca1e4a141b90282fa91c510b5ab05eabab9daa653a96a0df57b4edf044f.jpg)

![7101e96261d7c2676ac56adc1ddaa143e49e9eed22c1cb0c2965dbebb11465c7.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/7101e96261d7c2676ac56adc1ddaa143e49e9eed22c1cb0c2965dbebb11465c7.jpg)

![948d7bee64dc69764d8902d4a0b97d551054c9243e892bcbed69a1d985f6b180.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/948d7bee64dc69764d8902d4a0b97d551054c9243e892bcbed69a1d985f6b180.jpg)

![cf1e165cbf01c0e9b817309536a8d76a896f1f62d5bbfafdcf9473f612205833.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/cf1e165cbf01c0e9b817309536a8d76a896f1f62d5bbfafdcf9473f612205833.jpg)

![d02abb096e77018a9b8d8d737ea6ea7b55a61d52cbb5aa2b2c88863c450ead8d.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/d02abb096e77018a9b8d8d737ea6ea7b55a61d52cbb5aa2b2c88863c450ead8d.jpg)

![d5b6d8196d798520b24ab191ed55ea3f3416631643d7e3dd4ed6f8adf7e6bb74.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/d5b6d8196d798520b24ab191ed55ea3f3416631643d7e3dd4ed6f8adf7e6bb74.jpg)

![e60c77b9713f747d396c6cbdfcd3a22646afc8f81b8aff7a3cd4cea028bf822f.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/images/e60c77b9713f747d396c6cbdfcd3a22646afc8f81b8aff7a3cd4cea028bf822f.jpg)

### Tables

![1d34850b2fefe03534b65ab589ca9186b252c82ab4e2a2ab588dea3b3c511c24.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/1d34850b2fefe03534b65ab589ca9186b252c82ab4e2a2ab588dea3b3c511c24.jpg)

![26b68780df479dc2ff86185b18275ff969c28b84f5da10daceef64b4cc6f66a0.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/26b68780df479dc2ff86185b18275ff969c28b84f5da10daceef64b4cc6f66a0.jpg)

![2f1f0421f998522cda46ea6aefeee233e679742ad3be4b134a07917a18a798e6.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/2f1f0421f998522cda46ea6aefeee233e679742ad3be4b134a07917a18a798e6.jpg)

![452c6c9c6a7242e60798c42365ab29fc0c95fb418798616b8495f8b50fd2e0f1.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/452c6c9c6a7242e60798c42365ab29fc0c95fb418798616b8495f8b50fd2e0f1.jpg)

![47e1e1f91452d7666620a388159649338d397e90a575f7fe2f4b07ce9e9ccdb4.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/47e1e1f91452d7666620a388159649338d397e90a575f7fe2f4b07ce9e9ccdb4.jpg)

![7b20143ae0ea82828869aa914767d44ee68464a1fbaebbb695ee70df22cf0960.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/7b20143ae0ea82828869aa914767d44ee68464a1fbaebbb695ee70df22cf0960.jpg)

![802f97807470ecf7458f53b9d6f4bfa4b6233accaf68cb27fc99c36fbe68d72a.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/802f97807470ecf7458f53b9d6f4bfa4b6233accaf68cb27fc99c36fbe68d72a.jpg)

![84c93f103ac2c81ce7cf362e7d8c20f4d3515bce372c319d07c0ee7288e125c0.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/84c93f103ac2c81ce7cf362e7d8c20f4d3515bce372c319d07c0ee7288e125c0.jpg)

![8bbb5fce0cd1e6a4d6d3098dd4dcffea4f7c54d559f8c3d0fb46593c7edd965d.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/8bbb5fce0cd1e6a4d6d3098dd4dcffea4f7c54d559f8c3d0fb46593c7edd965d.jpg)

![9d2ad05fa9ae844a82fea2d31a201204862b6d2fa8d615bd8ed4a75fbf89c9e9.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/9d2ad05fa9ae844a82fea2d31a201204862b6d2fa8d615bd8ed4a75fbf89c9e9.jpg)

![aeff04730a76e85ed975bedd1bf55f5157b51374fc7c2dff7c9068446b245daf.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/aeff04730a76e85ed975bedd1bf55f5157b51374fc7c2dff7c9068446b245daf.jpg)

![eb22e2153c625a159e4628987ce8da85fa19e8918316a5c938b88437ed665301.jpg](../icml_results/1041_How%20can%20we%20assess%20human-agent%20interactions_%20Case%20studies%20in%20software%20agent%20design/tables/eb22e2153c625a159e4628987ce8da85fa19e8918316a5c938b88437ed665301.jpg)

## Adaptive Estimation and Inference in Semi-parametric Heterogeneous Clustered Multitask Learning via Neyman Orthogonality


### Images

![02cd738baa42572db552ebcf9cb19f86a2b8b4f511fa5c876ae473af2c545ed7.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/images/02cd738baa42572db552ebcf9cb19f86a2b8b4f511fa5c876ae473af2c545ed7.jpg)

![6cb94289543e1ca7a191d29d031c184d8e8ffb7487284868663ffcebae53c34d.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/images/6cb94289543e1ca7a191d29d031c184d8e8ffb7487284868663ffcebae53c34d.jpg)

![91b638f031277f08c7e633e7fc397cf0cf568f1a985d5ed56034cf6c36f3867c.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/images/91b638f031277f08c7e633e7fc397cf0cf568f1a985d5ed56034cf6c36f3867c.jpg)

![b8bb22c043c60aecc413a401a35475fa4b128083874bb5db34b7c4ae90de7634.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/images/b8bb22c043c60aecc413a401a35475fa4b128083874bb5db34b7c4ae90de7634.jpg)

![c6aef60089ec51f178cef9ef48eb3261f6059c689b0dc61e401da013b6b9a959.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/images/c6aef60089ec51f178cef9ef48eb3261f6059c689b0dc61e401da013b6b9a959.jpg)

![fa3697580acd8c8ce790c0b9bbcb7b32ba7dce513c41226d1c91be9cba5eeeb4.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/images/fa3697580acd8c8ce790c0b9bbcb7b32ba7dce513c41226d1c91be9cba5eeeb4.jpg)

### Tables

![07b8f9ef5e61eea2fcd7136a9952dcc8ae42b24c3d82c711ec4b1773ab25d09a.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/tables/07b8f9ef5e61eea2fcd7136a9952dcc8ae42b24c3d82c711ec4b1773ab25d09a.jpg)

![744c72ae51f9784ff0fdcc4905bc761c1af4c4e6d76d1db04d9c134dcaf6c9bf.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/tables/744c72ae51f9784ff0fdcc4905bc761c1af4c4e6d76d1db04d9c134dcaf6c9bf.jpg)

![ee6569567f9d06c0e6ca6e778be2c776570b479d3330892ebb36fb68cd2f6ad1.jpg](../icml_results/1042_Adaptive%20Estimation%20and%20Inference%20in%20Semi-parametric%20Heterogeneous%20Clustered%20Multitask%20Learning%20via%20/tables/ee6569567f9d06c0e6ca6e778be2c776570b479d3330892ebb36fb68cd2f6ad1.jpg)

<a id="paper-1025"></a>

## Olmix: A Framework for Data Mixing Throughout LM Development


### Images

![00685c543d98cbf71f94558ea2e6126d35006f23396384846734d85a2901fcd8.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/00685c543d98cbf71f94558ea2e6126d35006f23396384846734d85a2901fcd8.jpg)

![025fa79aef7e2f88c8bd7ff36bb7be2a9dc92be64cbd7dea6a022dfeb9f3ddba.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/025fa79aef7e2f88c8bd7ff36bb7be2a9dc92be64cbd7dea6a022dfeb9f3ddba.jpg)

![03b5b2e6254c64f577a6ee0626187a9890aba5d86d7d008830c50d06f96d4e33.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/03b5b2e6254c64f577a6ee0626187a9890aba5d86d7d008830c50d06f96d4e33.jpg)

![08357bdcd474b7e3255106ad41246e4e4aa9a9ed9f7e8be34c553e4448052e14.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/08357bdcd474b7e3255106ad41246e4e4aa9a9ed9f7e8be34c553e4448052e14.jpg)

![089636b6cd7959321601033d8e7a6a4d88734c76d49dc885b3d30eca8fde1f6e.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/089636b6cd7959321601033d8e7a6a4d88734c76d49dc885b3d30eca8fde1f6e.jpg)

![0b5c3b3b27151e92b6003ff6a274abd63c99a195c5764c9cec9bf3d85a51ee5c.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/0b5c3b3b27151e92b6003ff6a274abd63c99a195c5764c9cec9bf3d85a51ee5c.jpg)

![0bb04ef05ab7f7a5e994bea2291b70294560ac226c668095e587b709b47c2c0c.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/0bb04ef05ab7f7a5e994bea2291b70294560ac226c668095e587b709b47c2c0c.jpg)

![188f62524b2d3eae1d47a7dd8957f1271cf259a04ceb196cbc5dca0ffdd8139e.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/188f62524b2d3eae1d47a7dd8957f1271cf259a04ceb196cbc5dca0ffdd8139e.jpg)

![42cb0b0ed887a07d73e537ebfd6f50b7d91efc9267de8bfee0daba3dc4d02979.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/42cb0b0ed887a07d73e537ebfd6f50b7d91efc9267de8bfee0daba3dc4d02979.jpg)

![56673d97d0b889b923f62bd0b80093a4ceb2883ee204dc8470053ecb64339cd3.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/56673d97d0b889b923f62bd0b80093a4ceb2883ee204dc8470053ecb64339cd3.jpg)

![64def491dfdc6483e4ebb21a8138fb4f7b6c6c082e185ea99dc6860252bab045.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/64def491dfdc6483e4ebb21a8138fb4f7b6c6c082e185ea99dc6860252bab045.jpg)

![68230353319d0319cbdd4fc12180e6b127da148d9e133017f29d0c7eab91efbe.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/68230353319d0319cbdd4fc12180e6b127da148d9e133017f29d0c7eab91efbe.jpg)

![6ce281ef2d28bf2bd2c89d71325d35b27d45e99fe5e80dac4e844748d65fb826.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/6ce281ef2d28bf2bd2c89d71325d35b27d45e99fe5e80dac4e844748d65fb826.jpg)

![6d7163d5c0bca0465472c35782f0dada8d182f4d7ba7614701a9cd020fd13f1d.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/6d7163d5c0bca0465472c35782f0dada8d182f4d7ba7614701a9cd020fd13f1d.jpg)

![79d74678c209735f5fbbb57d54eff4b4ca001439d2f94dce532cac61489ad9ae.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/79d74678c209735f5fbbb57d54eff4b4ca001439d2f94dce532cac61489ad9ae.jpg)

![7a6d3cede8ee37f3721ce5cf783852b7a8af8f32233493b802541550e53be225.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/7a6d3cede8ee37f3721ce5cf783852b7a8af8f32233493b802541550e53be225.jpg)

![7ee50b296d10d6407a80a0e31be5c626b40c236557c55c2762b415fb5d31f9c3.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/7ee50b296d10d6407a80a0e31be5c626b40c236557c55c2762b415fb5d31f9c3.jpg)

![81cbc75d1dab9d61095750452573599ff631a21aa375e5794d361f06c3352dbf.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/81cbc75d1dab9d61095750452573599ff631a21aa375e5794d361f06c3352dbf.jpg)

![858dc78d6840d875b3628c55a563352a92e09b631400d00027348fc83e2b1b57.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/858dc78d6840d875b3628c55a563352a92e09b631400d00027348fc83e2b1b57.jpg)

![91aef781d10a57700100fa282b893c831b8eb689cf8137aaeb35aaaae15c7e31.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/91aef781d10a57700100fa282b893c831b8eb689cf8137aaeb35aaaae15c7e31.jpg)

![a2a8beed7381bc0425e4dfdb194c94c1bff77742d88895ad6afb1811733321e7.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/a2a8beed7381bc0425e4dfdb194c94c1bff77742d88895ad6afb1811733321e7.jpg)

![b7cf51950935ca57da658653971d026ab04f528c68410b4e26682e91936e060a.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/b7cf51950935ca57da658653971d026ab04f528c68410b4e26682e91936e060a.jpg)

![b9b1e1b665b8123e42bde78058bd80cb62767d815bf8718984858db273125a59.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/b9b1e1b665b8123e42bde78058bd80cb62767d815bf8718984858db273125a59.jpg)

![c7e1b3b28fbabd83198a30d709a1009e9822b8fdfb307bb0a509eb0477bf6948.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/c7e1b3b28fbabd83198a30d709a1009e9822b8fdfb307bb0a509eb0477bf6948.jpg)

![cac0168c60df2af0547ee7546d7003ac58d322d874f2e20707a700af68c98744.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/cac0168c60df2af0547ee7546d7003ac58d322d874f2e20707a700af68c98744.jpg)

![cac5ae8c946a00a0d03c044c16fedc42d1e0bd6bf075239baf0579b80faf8b2e.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/cac5ae8c946a00a0d03c044c16fedc42d1e0bd6bf075239baf0579b80faf8b2e.jpg)

![caf2208a24ad8304999942c7406df993f45b063f469f8b9fab6848093256c143.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/caf2208a24ad8304999942c7406df993f45b063f469f8b9fab6848093256c143.jpg)

![cce0ed3197d9f112d2e9452084a224f05d558024cedc0e0b09389de4b0b5fdb1.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/cce0ed3197d9f112d2e9452084a224f05d558024cedc0e0b09389de4b0b5fdb1.jpg)

![dbd5b9f915d0bb30a95e052179b88d236846ef79a3b9d002646d794e3341ef47.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/images/dbd5b9f915d0bb30a95e052179b88d236846ef79a3b9d002646d794e3341ef47.jpg)

### Tables

![01490ae8a18c459ce103aaca85e48c4f979bd763865f43969d9f4400b835ec2f.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/01490ae8a18c459ce103aaca85e48c4f979bd763865f43969d9f4400b835ec2f.jpg)

![26218c20f01205cdb6a1e955cf040168b39abc0bcc1feb60dc4087de41c8e333.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/26218c20f01205cdb6a1e955cf040168b39abc0bcc1feb60dc4087de41c8e333.jpg)

![2aa3952efb52bd00adab2b7525195607fe8cfa98bc6c9b521873f5fd0aa7f919.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/2aa3952efb52bd00adab2b7525195607fe8cfa98bc6c9b521873f5fd0aa7f919.jpg)

![37994ba73398e530e833c738af16ece7454ad943d71aa301745c47c60dc1fcd7.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/37994ba73398e530e833c738af16ece7454ad943d71aa301745c47c60dc1fcd7.jpg)

![38572956a90afaf2edfc6efdc800c1d893a363fcbede096887267d88628b28fa.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/38572956a90afaf2edfc6efdc800c1d893a363fcbede096887267d88628b28fa.jpg)

![5824d3c2c048c197fdfab22444db645555be74453125e528843446ca5f7c134c.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/5824d3c2c048c197fdfab22444db645555be74453125e528843446ca5f7c134c.jpg)

![7a379e568461ac17580bdb2dc9bbda06227800683d028f33a549a13e221a6fe4.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/7a379e568461ac17580bdb2dc9bbda06227800683d028f33a549a13e221a6fe4.jpg)

![ab65bf56af74db7d582e75375382db1a45931c17d17d215520629baaa0285ed3.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/ab65bf56af74db7d582e75375382db1a45931c17d17d215520629baaa0285ed3.jpg)

![c9f4c818ad07605c6a1a99d4908e2562dcf1b4825012a9c1cde671cdddee00c9.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/c9f4c818ad07605c6a1a99d4908e2562dcf1b4825012a9c1cde671cdddee00c9.jpg)

![d2773e6bfd95878251da23cbc83041d4be5d4ccc2775299d45e3774513cfcb1a.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/d2773e6bfd95878251da23cbc83041d4be5d4ccc2775299d45e3774513cfcb1a.jpg)

![d4d0e241345bc3ef3a1b96dc678b55e56a23100318612c023749d19bf1585dd7.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/d4d0e241345bc3ef3a1b96dc678b55e56a23100318612c023749d19bf1585dd7.jpg)

![e6cc6e2ebdea17df1ff77935577897886c588aa33a9c1691b6116bdde320bcdb.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/e6cc6e2ebdea17df1ff77935577897886c588aa33a9c1691b6116bdde320bcdb.jpg)

![eaf385b92468f75970d4adadd71b14ffec85dac75e7d931eeee6e856f3fa71e5.jpg](../icml_results/1043_Olmix_%20A%20Framework%20for%20Data%20Mixing%20Throughout%20LM%20Development/tables/eaf385b92468f75970d4adadd71b14ffec85dac75e7d931eeee6e856f3fa71e5.jpg)

<a id="paper-1026"></a>

## How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning


### Images

![00220ef06d51a2cf6892a8dd3360d55a94cd5f1843b86c999d8ee1e0e51b00c9.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/00220ef06d51a2cf6892a8dd3360d55a94cd5f1843b86c999d8ee1e0e51b00c9.jpg)

![058cba7ec1d764bc5e09c4b942e2967e8666c9efb340cd595aadae9d88f8af89.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/058cba7ec1d764bc5e09c4b942e2967e8666c9efb340cd595aadae9d88f8af89.jpg)

![102de5be7d07d8fb93315b99b5f9fbc7692d62ff3af9dca9e5648303bba9d0a0.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/102de5be7d07d8fb93315b99b5f9fbc7692d62ff3af9dca9e5648303bba9d0a0.jpg)

![2cf6efbab24a28e1bbf3b9926a5015228641e7678d0ee0d2e0eede7a5a3015b3.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/2cf6efbab24a28e1bbf3b9926a5015228641e7678d0ee0d2e0eede7a5a3015b3.jpg)

![5c1f4fffa7054c206e9c1c9199b53bce2714333258f548737f2ae5394f36e077.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/5c1f4fffa7054c206e9c1c9199b53bce2714333258f548737f2ae5394f36e077.jpg)

![a2eb84374ffbf06a77512c0bf288458b15991bffdab31cef793b97fbe2f6ba64.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/a2eb84374ffbf06a77512c0bf288458b15991bffdab31cef793b97fbe2f6ba64.jpg)

![abab478f646e932da94cbd5aceb5f00ea1000658c8fa1e4dd8844f70f661f3de.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/abab478f646e932da94cbd5aceb5f00ea1000658c8fa1e4dd8844f70f661f3de.jpg)

![b274b16afe319fcdc2727a9688d2516d63c13258f1e1266b19aa50891002b828.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/b274b16afe319fcdc2727a9688d2516d63c13258f1e1266b19aa50891002b828.jpg)

![c813c3d8b1ae9ab7e240074336687b1be90a3dda30fb3280557b90f2016ecfb0.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/c813c3d8b1ae9ab7e240074336687b1be90a3dda30fb3280557b90f2016ecfb0.jpg)

![d4252333e70c0cc78150aa2fbb8557279857d23a82a96a5cf46c3dac35a7a222.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/d4252333e70c0cc78150aa2fbb8557279857d23a82a96a5cf46c3dac35a7a222.jpg)

![e162ebdf8565ea5b492ba510e024cd7ae5d026833b571795a126158d5b90a80a.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/e162ebdf8565ea5b492ba510e024cd7ae5d026833b571795a126158d5b90a80a.jpg)

![ea7be2f3946c97b5de09c218205371d85a84676536d8abcdbafffea075613dfa.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/ea7be2f3946c97b5de09c218205371d85a84676536d8abcdbafffea075613dfa.jpg)

![fa78f6dd50e70ccaea912d4b3ca6a3e0a98388608634488d9ceb65dc1deef840.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/images/fa78f6dd50e70ccaea912d4b3ca6a3e0a98388608634488d9ceb65dc1deef840.jpg)

### Tables

![64ffe51acc004757b11053e95da271e8eb08ffea2cd0481fe8ca0bf77ead0fe9.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/tables/64ffe51acc004757b11053e95da271e8eb08ffea2cd0481fe8ca0bf77ead0fe9.jpg)

![6fe19ca92c69d304d3004039ffb44c0de42d9e669603b8aff9667677d2d4311d.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/tables/6fe19ca92c69d304d3004039ffb44c0de42d9e669603b8aff9667677d2d4311d.jpg)

![8f1f18ca9c90f1ffacb978b581faabfd91e486a7d71d8e9aa6a05432ad334da2.jpg](../icml_results/1044_How%20Hard%20Can%20It%20Be_%20Hardness-Aware%20Multi-Objective%20Unlearning/tables/8f1f18ca9c90f1ffacb978b581faabfd91e486a7d71d8e9aa6a05432ad334da2.jpg)

<a id="paper-1027"></a>

## Smooth Multi-Policy Causal Effect Estimation in Longitudinal Settings


### Images

![21067f3474700012e91df39ecb65d51623426071d3def57a349a6761a0741d81.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/images/21067f3474700012e91df39ecb65d51623426071d3def57a349a6761a0741d81.jpg)

![8114984b7cc6e3defd51a0d0b48da037d7aae9a7f1d08ba9a3eacf8475e616a9.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/images/8114984b7cc6e3defd51a0d0b48da037d7aae9a7f1d08ba9a3eacf8475e616a9.jpg)

![90ea3931ea2a737ad83ee214aa9ce52f211785d27bb7dcb256d74ac979ad9029.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/images/90ea3931ea2a737ad83ee214aa9ce52f211785d27bb7dcb256d74ac979ad9029.jpg)

![a10cf56fc52576bfc728e1458dff6119af647979b3d9aa355e3fec9b56697f95.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/images/a10cf56fc52576bfc728e1458dff6119af647979b3d9aa355e3fec9b56697f95.jpg)

### Tables

![03090c98d4dc96f9f2d7f3cc71b17d84828cb989c80928a5e8db9968d951a30e.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/03090c98d4dc96f9f2d7f3cc71b17d84828cb989c80928a5e8db9968d951a30e.jpg)

![158c1fabcaaed588bd89a414aac8453345fc2fcd5b63582f8c0f06338b7b17c8.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/158c1fabcaaed588bd89a414aac8453345fc2fcd5b63582f8c0f06338b7b17c8.jpg)

![1a4fced337d43e2375204c5465608332606dda8802f84fb15d26e553b7966813.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/1a4fced337d43e2375204c5465608332606dda8802f84fb15d26e553b7966813.jpg)

![37cc130c44e110d452981322acb3653dfae8237d9309e83cdcf9af6917b356c4.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/37cc130c44e110d452981322acb3653dfae8237d9309e83cdcf9af6917b356c4.jpg)

![38e8658d936b387abfdca01abdbf688c9552bdbec9a2873451151aa7e897392b.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/38e8658d936b387abfdca01abdbf688c9552bdbec9a2873451151aa7e897392b.jpg)

![6644ff38dead4766b31ec6f78278939bbeef5a131d2ba3f7e8a7a49102e1999a.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/6644ff38dead4766b31ec6f78278939bbeef5a131d2ba3f7e8a7a49102e1999a.jpg)

![6febcda9fe6189c0d7e2b686491064a534f43dae4ffde32fbeb331d249b1cc64.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/6febcda9fe6189c0d7e2b686491064a534f43dae4ffde32fbeb331d249b1cc64.jpg)

![aa5b121d93c351e4a80d40beb8cdcaa8961d2446062983a4b22f88d2681a2c47.jpg](../icml_results/1045_Smooth%20Multi-Policy%20Causal%20Effect%20Estimation%20in%20Longitudinal%20Settings/tables/aa5b121d93c351e4a80d40beb8cdcaa8961d2446062983a4b22f88d2681a2c47.jpg)

<a id="paper-1028"></a>

## Discovering Symmetry Groups with Flow Matching


### Images

![00c35942a759886f7dfd1ccbac9f6815847e2088df4694cfb6af8ce445b344e7.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/00c35942a759886f7dfd1ccbac9f6815847e2088df4694cfb6af8ce445b344e7.jpg)

![0221266e5c5f4850798c9eb966a7407adbfe373d584caf54a499931dd4ec2e1c.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/0221266e5c5f4850798c9eb966a7407adbfe373d584caf54a499931dd4ec2e1c.jpg)

![10e407ba4fce1ce9bca133e25daac9c77713abf0177455cb40990eabd39def7b.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/10e407ba4fce1ce9bca133e25daac9c77713abf0177455cb40990eabd39def7b.jpg)

![174bd26a0072426dc357695c9abdae0b731d2b57853fb00e3beb1fd97e4c45c7.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/174bd26a0072426dc357695c9abdae0b731d2b57853fb00e3beb1fd97e4c45c7.jpg)

![196915d379c3fdbf05efd3cf641a27a87f9c7ae4d235740d0a0e6f553543444a.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/196915d379c3fdbf05efd3cf641a27a87f9c7ae4d235740d0a0e6f553543444a.jpg)

![1a8d50c818bb10db153311eb7c347549b257364d0b18bcb2a4bdd548d678d8ce.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/1a8d50c818bb10db153311eb7c347549b257364d0b18bcb2a4bdd548d678d8ce.jpg)

![22e7d8540e7a0c768d350bbb146e940e5d6d9ae1aa9cd8e1b510746828ada371.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/22e7d8540e7a0c768d350bbb146e940e5d6d9ae1aa9cd8e1b510746828ada371.jpg)

![273876d446ac6f08c4424fab3f3901977a58807545cd668c706bc0546b3d969e.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/273876d446ac6f08c4424fab3f3901977a58807545cd668c706bc0546b3d969e.jpg)

![2f6de066fb668f973a7d0eeaaaf2a935d3937d94b0806a5b716162d9812a2336.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/2f6de066fb668f973a7d0eeaaaf2a935d3937d94b0806a5b716162d9812a2336.jpg)

![387504024467bdcd51c84a7ff478ea8d93202f06c4262da7a27711787ecf9a4e.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/387504024467bdcd51c84a7ff478ea8d93202f06c4262da7a27711787ecf9a4e.jpg)

![39874931db9c5aa780c30325ab5f4a80358450cf08d017fd4f455bf3aa888905.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/39874931db9c5aa780c30325ab5f4a80358450cf08d017fd4f455bf3aa888905.jpg)

![3a744a6a72d56e6f513e44c4329060f5bad401d6af5b94255708fe70a4752005.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/3a744a6a72d56e6f513e44c4329060f5bad401d6af5b94255708fe70a4752005.jpg)

![3d60d21025c7eff603aafb3daf6b3f5b4cadf57f27fb4425cb21b90c863d972d.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/3d60d21025c7eff603aafb3daf6b3f5b4cadf57f27fb4425cb21b90c863d972d.jpg)

![3e3cf422686bf67b8a0a85daeaa5f3bc96c72c5bb5d7b84b234e572e003effe0.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/3e3cf422686bf67b8a0a85daeaa5f3bc96c72c5bb5d7b84b234e572e003effe0.jpg)

![48991a48b8678c0f8e95ac3ceee7fa1eaae1e0641dc489c761938c6f95ae36b2.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/48991a48b8678c0f8e95ac3ceee7fa1eaae1e0641dc489c761938c6f95ae36b2.jpg)

![4aa930c7ece443663bc7f58de2b560d38a3fe31fcc09e252675a41004e1f441c.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/4aa930c7ece443663bc7f58de2b560d38a3fe31fcc09e252675a41004e1f441c.jpg)

![55fe6e942898b9556980e728d907bfd25eb8f5a455e53e19a1fadb67f102d162.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/55fe6e942898b9556980e728d907bfd25eb8f5a455e53e19a1fadb67f102d162.jpg)

![6ba283eb5781170eaac11178f3330a9aa99603c2092a550f44e3501b5ce1c2cf.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/6ba283eb5781170eaac11178f3330a9aa99603c2092a550f44e3501b5ce1c2cf.jpg)

![6bd71b50d0fc3ffda9df5b96db2b179e8c9092e297db0f021267b00fede80ba4.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/6bd71b50d0fc3ffda9df5b96db2b179e8c9092e297db0f021267b00fede80ba4.jpg)

![74b76e73e78759d8d1542289e52c6151ac316249a068c75c0f6c283b16cb9569.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/74b76e73e78759d8d1542289e52c6151ac316249a068c75c0f6c283b16cb9569.jpg)

![83e5bbd67e862c1f6a822d26a607cee7b6c6a7630991d1dc1dc7b2eb005da623.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/83e5bbd67e862c1f6a822d26a607cee7b6c6a7630991d1dc1dc7b2eb005da623.jpg)

![85e4f0ce8d4def7776307cc1acecc2047c30019a86880d3c51eea08d0784d1f7.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/85e4f0ce8d4def7776307cc1acecc2047c30019a86880d3c51eea08d0784d1f7.jpg)

![8612f253f4bcc0149d6b04fb4ab5a7686e7cf92349f100719e9a32b1d65e3e66.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/8612f253f4bcc0149d6b04fb4ab5a7686e7cf92349f100719e9a32b1d65e3e66.jpg)

![96b031888b28ee37b5e9a8e5b19e643ce568fce54ca5a943e64d07a20ba6394c.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/96b031888b28ee37b5e9a8e5b19e643ce568fce54ca5a943e64d07a20ba6394c.jpg)

![b09c95fb19a0303ecadf4937b7f5a5222fe33b7c5812873c17deee6a8b71189e.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/b09c95fb19a0303ecadf4937b7f5a5222fe33b7c5812873c17deee6a8b71189e.jpg)

![b1df278726fbd610e0bbfe3831496838ccfb69cf56c8146069c2c2ed5992c8c6.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/b1df278726fbd610e0bbfe3831496838ccfb69cf56c8146069c2c2ed5992c8c6.jpg)

![bbf5a74bafcc5b201183c66e717697cb7669e65a1822701abd1a72e98f56736f.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/bbf5a74bafcc5b201183c66e717697cb7669e65a1822701abd1a72e98f56736f.jpg)

![bd6d14d8c09371603a3a8307ca08164c867784bda7e6fdafc9e6265407c81e99.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/bd6d14d8c09371603a3a8307ca08164c867784bda7e6fdafc9e6265407c81e99.jpg)

![ca5d9d74b4ebbd924fc703b80918695cadb39c0542e59f977eadda3d36ab0cbe.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/ca5d9d74b4ebbd924fc703b80918695cadb39c0542e59f977eadda3d36ab0cbe.jpg)

![d0d2c7b58bb764d505c95f343759f7952f96d22688fda20bdae5db627139d731.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/d0d2c7b58bb764d505c95f343759f7952f96d22688fda20bdae5db627139d731.jpg)

![dbfb9afdc9d4ab6631aa3733d1117a55e1261cdb0fcbb2f64125e85552032db1.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/dbfb9afdc9d4ab6631aa3733d1117a55e1261cdb0fcbb2f64125e85552032db1.jpg)

![f391e566ed25f318030875aa6002a017f56c02f6dc805b60a659f859db3650d6.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/f391e566ed25f318030875aa6002a017f56c02f6dc805b60a659f859db3650d6.jpg)

![f85612bcf29d7b87486d6617fe686916b99e4abdd2dd5e23c8de0f685be37ae1.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/f85612bcf29d7b87486d6617fe686916b99e4abdd2dd5e23c8de0f685be37ae1.jpg)

![ff4429d00c240d6e43307861cdfc0e8533dbfc2a32410a34f1a307fece2af033.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/ff4429d00c240d6e43307861cdfc0e8533dbfc2a32410a34f1a307fece2af033.jpg)

![ff4849ad1d27de139f85c115344699e3acb453f4904ddd52615c3e6aa953f017.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/images/ff4849ad1d27de139f85c115344699e3acb453f4904ddd52615c3e6aa953f017.jpg)

### Tables

![010975b2b8985a3124d445f8d72fe431832383aa1e5c015f7f79cb6b03b4ae74.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/tables/010975b2b8985a3124d445f8d72fe431832383aa1e5c015f7f79cb6b03b4ae74.jpg)

![9df9fde1d7b8edc3538619f45eb719bdd668bf2a2e56b4fa0acad2f525ab1038.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/tables/9df9fde1d7b8edc3538619f45eb719bdd668bf2a2e56b4fa0acad2f525ab1038.jpg)

![a46f76fe0ed6b47788d0c10e3523a946d84cb693edfa27f4d1b0de531c400e5e.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/tables/a46f76fe0ed6b47788d0c10e3523a946d84cb693edfa27f4d1b0de531c400e5e.jpg)

![af2c46f52ecbc8d049b6fefd77aee7f3aa152a1e47f74aca0daac538c79fdd6a.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/tables/af2c46f52ecbc8d049b6fefd77aee7f3aa152a1e47f74aca0daac538c79fdd6a.jpg)

![ea8067723fa13aaccfc8831201ee8182ab847a038ac622a1b055bf7c9a4e211f.jpg](../icml_results/1046_Discovering%20Symmetry%20Groups%20with%20Flow%20Matching/tables/ea8067723fa13aaccfc8831201ee8182ab847a038ac622a1b055bf7c9a4e211f.jpg)

<a id="paper-1029"></a>

## MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs


### Images

![165dc1ca75b071e16d3dbcdd8db792d46cb0bc75b8eee8bc2094f5ad8d267aa1.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/165dc1ca75b071e16d3dbcdd8db792d46cb0bc75b8eee8bc2094f5ad8d267aa1.jpg)

![298bdf5e2b07bb0472f5c0971bb4386ae14bb04b9cfa7db6bdbc8c0d973b8cb4.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/298bdf5e2b07bb0472f5c0971bb4386ae14bb04b9cfa7db6bdbc8c0d973b8cb4.jpg)

![2d77f01f6ce5cca2ed1b9f74e884b8cfa53ce300a97e85e6337b6410d8361598.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/2d77f01f6ce5cca2ed1b9f74e884b8cfa53ce300a97e85e6337b6410d8361598.jpg)

![33c8581450fd8c5df92f92e2c5a4662a3ef612cdb45838e7a610fc4d17f57084.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/33c8581450fd8c5df92f92e2c5a4662a3ef612cdb45838e7a610fc4d17f57084.jpg)

![34eb133765fb3fe0509dc014b987dbddc822496c17f592eca9ea70f86338bedb.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/34eb133765fb3fe0509dc014b987dbddc822496c17f592eca9ea70f86338bedb.jpg)

![492d64b958f5d8e5546daef5c3f5f57dc7cfcb071c9f34fa87a233f12c34070e.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/492d64b958f5d8e5546daef5c3f5f57dc7cfcb071c9f34fa87a233f12c34070e.jpg)

![4eb07475d711559e9253da3b0b227a6cff0b59ac84d1f276d3d9c1f3fe377c87.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/4eb07475d711559e9253da3b0b227a6cff0b59ac84d1f276d3d9c1f3fe377c87.jpg)

![53cc902145d0ba7c7f4e10e4ee71d8da0bad122b62d3caaceaea05f296bba808.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/53cc902145d0ba7c7f4e10e4ee71d8da0bad122b62d3caaceaea05f296bba808.jpg)

![a11a316c74378a500bc70a50259aa43909b7af1e84d6c8b00a4bf66a0c24389d.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/a11a316c74378a500bc70a50259aa43909b7af1e84d6c8b00a4bf66a0c24389d.jpg)

![a8a2b5a585d8e478aebc42238471839fd6e00798f43e20c62de9dca2849a18ff.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/a8a2b5a585d8e478aebc42238471839fd6e00798f43e20c62de9dca2849a18ff.jpg)

![b2b6da19331637c55256965afd191f5bebc398fc3ce025d765b3f42fb0fc1d97.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/images/b2b6da19331637c55256965afd191f5bebc398fc3ce025d765b3f42fb0fc1d97.jpg)

### Tables

![5944daff2ffbc8dec417ddf136231bdfd0979ce25b1c9b08a1a2b8b94e21ec08.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/tables/5944daff2ffbc8dec417ddf136231bdfd0979ce25b1c9b08a1a2b8b94e21ec08.jpg)

![6c417e2bea36a3a4ce2bb3453ae71587ef8fb33a9cc12a979819939a763862e0.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/tables/6c417e2bea36a3a4ce2bb3453ae71587ef8fb33a9cc12a979819939a763862e0.jpg)

![85899719e21ebe26811a659abfb22ed21018e1886dd2fbe0b3a7eae5cc8cfd70.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/tables/85899719e21ebe26811a659abfb22ed21018e1886dd2fbe0b3a7eae5cc8cfd70.jpg)

![c6883cdc038c371f5cf8d0f73db3e95e99adcf566e5f6a68996ce7b7dc3414cd.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/tables/c6883cdc038c371f5cf8d0f73db3e95e99adcf566e5f6a68996ce7b7dc3414cd.jpg)

![d8b0879b4d9f4aadbda386f5648b427d57535cf182fb136364276a22f5d67230.jpg](../icml_results/1047_MVI-Bench_%20A%20Comprehensive%20Benchmark%20for%20Evaluating%20Robustness%20to%20Misleading%20Visual%20Inputs%20in%20LVLMs/tables/d8b0879b4d9f4aadbda386f5648b427d57535cf182fb136364276a22f5d67230.jpg)

<a id="paper-1030"></a>

## Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens


### Images

![06b5b6589643ac65d7c16cccde92065ce62684cd77a42c684798cd2d3906eea9.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/06b5b6589643ac65d7c16cccde92065ce62684cd77a42c684798cd2d3906eea9.jpg)

![1410452a118de78cd3e3c4493dbcfba9dd834856f50b7ee5cf6baf20d17125bf.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/1410452a118de78cd3e3c4493dbcfba9dd834856f50b7ee5cf6baf20d17125bf.jpg)

![725a8c70f439062f73a2129a34d35ce211941651ba6a61493d4107362400d9da.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/725a8c70f439062f73a2129a34d35ce211941651ba6a61493d4107362400d9da.jpg)

![8976f31a58ecaf2ed387a8cb280115587f0b592999fea3a8da34f2610916104c.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/8976f31a58ecaf2ed387a8cb280115587f0b592999fea3a8da34f2610916104c.jpg)

![8f6c09266009c85cde874bd43fdd852d63b3c13f7b5718356e00722e9bc9ef16.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/8f6c09266009c85cde874bd43fdd852d63b3c13f7b5718356e00722e9bc9ef16.jpg)

![95faed6fbbb639aeae0eae01451717aa8187b8e687079775cd5139d2360e0dce.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/95faed6fbbb639aeae0eae01451717aa8187b8e687079775cd5139d2360e0dce.jpg)

![b3e081d495dd18d5fac3836be969e764b9be689215f1e0a4f98959d7d4e6395f.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/b3e081d495dd18d5fac3836be969e764b9be689215f1e0a4f98959d7d4e6395f.jpg)

![b7f6c2b1fbb6c144d9bae89e22f275ffee4150594c07508423e4a4b3fe6a578f.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/b7f6c2b1fbb6c144d9bae89e22f275ffee4150594c07508423e4a4b3fe6a578f.jpg)

![ca8d0f1ca74b96bbed7f2fd566e8dd33bfcfb22c264dbd71edecf9c6d11ed584.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/images/ca8d0f1ca74b96bbed7f2fd566e8dd33bfcfb22c264dbd71edecf9c6d11ed584.jpg)

### Tables

![76c1dff9a585510e75be1af897971a3347cb85b1c6413905af2db74ecc8ce782.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/tables/76c1dff9a585510e75be1af897971a3347cb85b1c6413905af2db74ecc8ce782.jpg)

![9ee12f3121d7a0f38992010677d1758be78b76272e152e93962c533e004d7313.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/tables/9ee12f3121d7a0f38992010677d1758be78b76272e152e93962c533e004d7313.jpg)

![f75e005f771a8176fed326e41b765c988c939a2b5ed476fd1a956bf608148f71.jpg](../icml_results/1048_Think%20Deep%2C%20Not%20Just%20Long_%20Measuring%20LLM%20Reasoning%20Effort%20via%20Deep-Thinking%20Tokens/tables/f75e005f771a8176fed326e41b765c988c939a2b5ed476fd1a956bf608148f71.jpg)

<a id="paper-1031"></a>

## SimpleGPT: Improving GPT via A Simple Normalization Strategy


### Images

![205c905b79ca52c5ea14fa9a005ca8bf7f9fbbc968ad2148621f0f2f0621c9cd.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/205c905b79ca52c5ea14fa9a005ca8bf7f9fbbc968ad2148621f0f2f0621c9cd.jpg)

![25f6be0f4df7cae30b82414f51b56c18a22ffe787308553c4cd3bad8a5bd28e0.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/25f6be0f4df7cae30b82414f51b56c18a22ffe787308553c4cd3bad8a5bd28e0.jpg)

![41eb00c32423eebfb0bdee9c358e410e2a8357828ff6bf79446835e8600f5d98.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/41eb00c32423eebfb0bdee9c358e410e2a8357828ff6bf79446835e8600f5d98.jpg)

![44de3c0ad3d61cd3d0e5db77510c8bcca42a479570e0b25d2d3c3a00270841d1.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/44de3c0ad3d61cd3d0e5db77510c8bcca42a479570e0b25d2d3c3a00270841d1.jpg)

![63e7c4c05b2c604a817074c9761660d9fdc64cdde94195fa694b89611f04a91a.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/63e7c4c05b2c604a817074c9761660d9fdc64cdde94195fa694b89611f04a91a.jpg)

![6fc83ee872f0d3723b78c7b68e581fe59ea9772dfcf8d8c25914c46e1b613a02.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/6fc83ee872f0d3723b78c7b68e581fe59ea9772dfcf8d8c25914c46e1b613a02.jpg)

![8a58d96a7308376590ab8ff897efc0a65649af7fd2468e09d7b5b5d5614d7b3a.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/8a58d96a7308376590ab8ff897efc0a65649af7fd2468e09d7b5b5d5614d7b3a.jpg)

![a30f489474f14a1c51282923e46bdf84c4f7d90d8bdf35ac2855ad2756cb69be.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/a30f489474f14a1c51282923e46bdf84c4f7d90d8bdf35ac2855ad2756cb69be.jpg)

![c06f370ecd058d23279b876716343bcafc92dcb1e7fddd1576ffd3f9d3318a84.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/c06f370ecd058d23279b876716343bcafc92dcb1e7fddd1576ffd3f9d3318a84.jpg)

![c1b4ab402d7ba3c134376fc4804d1d22da8926853631f3f6f4993649d54b5ea5.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/c1b4ab402d7ba3c134376fc4804d1d22da8926853631f3f6f4993649d54b5ea5.jpg)

![e228e1409b085fd5210a1cf0d346380d8d28bae0f3eeb155e7a884c493dc56ae.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/e228e1409b085fd5210a1cf0d346380d8d28bae0f3eeb155e7a884c493dc56ae.jpg)

![ea6dff4a6e3e47f546c51eade068cfab803e9441b154ed4c6c2e0e681a22f8f9.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/ea6dff4a6e3e47f546c51eade068cfab803e9441b154ed4c6c2e0e681a22f8f9.jpg)

![ef0c5affaebdd42cc571b3a8d6f4baa22d3ae2e08c23c67eaf8de489cad16b5a.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/ef0c5affaebdd42cc571b3a8d6f4baa22d3ae2e08c23c67eaf8de489cad16b5a.jpg)

![f041ecd8ecca39d61b2a46bcdfd5bacca5e7dc1859b30667c792146049409455.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/f041ecd8ecca39d61b2a46bcdfd5bacca5e7dc1859b30667c792146049409455.jpg)

![f702661512134c55789a11a547afd007fb05fa9c0048a5c5731d7f1b45b3acee.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/images/f702661512134c55789a11a547afd007fb05fa9c0048a5c5731d7f1b45b3acee.jpg)

### Tables

![3628ddb4afc9e1bf855639031751828454ba2e1d7ce5c179dd2d68b46d37f803.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/tables/3628ddb4afc9e1bf855639031751828454ba2e1d7ce5c179dd2d68b46d37f803.jpg)

![482b423004852b575c100ff8c9d00d56d0731372df0d1c1ce6d7084008b19ee7.jpg](../icml_results/1049_SimpleGPT_%20Improving%20GPT%20via%20A%20Simple%20Normalization%20Strategy/tables/482b423004852b575c100ff8c9d00d56d0731372df0d1c1ce6d7084008b19ee7.jpg)

<a id="paper-1032"></a>

## Multi-Agent Reinforcement Learning with Submodular Reward


### Images

![caecb708a58af81a677f60d24efcbd8cdbe1f638b13612a21b503f10a1cd438b.jpg](../icml_results/1050_Multi-Agent%20Reinforcement%20Learning%20with%20Submodular%20Reward/images/caecb708a58af81a677f60d24efcbd8cdbe1f638b13612a21b503f10a1cd438b.jpg)

<a id="paper-1033"></a>

## 3DGS-HPC: Distractor-free 3D Gaussian Splatting with Hybrid Patch-wise Classification


### Images

![177979e2413e4510bd75a17d2fea6cf96c6f58736c344758398c0fa19af21dad.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/177979e2413e4510bd75a17d2fea6cf96c6f58736c344758398c0fa19af21dad.jpg)

![4d11b60dbb5444343c4014321800f22e93b6cb6cb6e517336dc48a112e57bb9a.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/4d11b60dbb5444343c4014321800f22e93b6cb6cb6e517336dc48a112e57bb9a.jpg)

![53376cb9aab4e11394ad3a47a7eacd88d3e17ad2574895147d5ad884338e7b7a.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/53376cb9aab4e11394ad3a47a7eacd88d3e17ad2574895147d5ad884338e7b7a.jpg)

![54b962873b0cedb4124fd60d9f28feeaff67331e4f20cd71fd7e2f2ea9f1c243.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/54b962873b0cedb4124fd60d9f28feeaff67331e4f20cd71fd7e2f2ea9f1c243.jpg)

![60a56c897a00c70fd690858b4553194dec50347091f29e6840d0f19a3ab65996.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/60a56c897a00c70fd690858b4553194dec50347091f29e6840d0f19a3ab65996.jpg)

![65f384d2303c83b0691d4ffb06c67c28c80e757790313da624f5b1fa09ef6136.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/65f384d2303c83b0691d4ffb06c67c28c80e757790313da624f5b1fa09ef6136.jpg)

![73877338fe77d7c7c66ea3b7978e2289f84550413351f5ca9b6ce84269607d49.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/73877338fe77d7c7c66ea3b7978e2289f84550413351f5ca9b6ce84269607d49.jpg)

![7c417891b8e7dcb962e679071d7683890c22a6981e39462f18bff0dc6634053b.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/7c417891b8e7dcb962e679071d7683890c22a6981e39462f18bff0dc6634053b.jpg)

![841c35c4f7bf3e2201d9344c5cb58467617e78ec07b01407d5737a0ac9b9e5a8.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/841c35c4f7bf3e2201d9344c5cb58467617e78ec07b01407d5737a0ac9b9e5a8.jpg)

![9663524c13a07d5dcff2b419bc7d861572fb241c2560294a63090ccff80a95e0.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/9663524c13a07d5dcff2b419bc7d861572fb241c2560294a63090ccff80a95e0.jpg)

![97df3a275e815b25cb97d65d6757eeb39f0ba687e91d319c4bca085e7825d195.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/97df3a275e815b25cb97d65d6757eeb39f0ba687e91d319c4bca085e7825d195.jpg)

![afedbf194b57e628e3b767e6a2d0baebd907312cae8256023f527b1129feabad.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/afedbf194b57e628e3b767e6a2d0baebd907312cae8256023f527b1129feabad.jpg)

![c2f1fdf248192c1bbcb7cf8d5e967b9c353060569ebecc05e866ba01ef0d564b.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/c2f1fdf248192c1bbcb7cf8d5e967b9c353060569ebecc05e866ba01ef0d564b.jpg)

![d2b556692775cc1fc0e324458b33953d867752c177b33b5575ebd865f461b1ff.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/d2b556692775cc1fc0e324458b33953d867752c177b33b5575ebd865f461b1ff.jpg)

![e083ae79d3350f22db5eeb87c197655d348a339568aa9bee0e4379613df4e121.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/e083ae79d3350f22db5eeb87c197655d348a339568aa9bee0e4379613df4e121.jpg)

![e528c0f7ee9854bf7a0762c9b50666f447d99dfccb418b13bb8c2f6ad354784f.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/e528c0f7ee9854bf7a0762c9b50666f447d99dfccb418b13bb8c2f6ad354784f.jpg)

![f3406a5d1c904f8790e2401621773974463d5f72e5d919f56e3b8d6cf95bb1b7.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/images/f3406a5d1c904f8790e2401621773974463d5f72e5d919f56e3b8d6cf95bb1b7.jpg)

### Tables

![05be770f8bac0b3fe17edb13c6d6e23ac7d0016f6d89334d90d7edf585f3653c.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/tables/05be770f8bac0b3fe17edb13c6d6e23ac7d0016f6d89334d90d7edf585f3653c.jpg)

![5c2a41ea2d93fb5ef1a6eeef0d5d616dabe50eab99c9d6b10abd677ce10adcee.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/tables/5c2a41ea2d93fb5ef1a6eeef0d5d616dabe50eab99c9d6b10abd677ce10adcee.jpg)

![7dc08fb6340899561cc237e848774c23ea1aacaefe9e92885076b0468fa750f7.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/tables/7dc08fb6340899561cc237e848774c23ea1aacaefe9e92885076b0468fa750f7.jpg)

![a1cb61091e43e318f9b63f44cdab4f3897d222b461d3a80ba291d7c9076d4421.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/tables/a1cb61091e43e318f9b63f44cdab4f3897d222b461d3a80ba291d7c9076d4421.jpg)

![d347b1da9b24945a6dc1819d0ac6c24a22e2e62e041256aeb5f7f81f2b8e7af5.jpg](../icml_results/1051_3DGS-HPC_%20Distractor-free%203D%20Gaussian%20Splatting%20with%20Hybrid%20Patch-wise%20Classification/tables/d347b1da9b24945a6dc1819d0ac6c24a22e2e62e041256aeb5f7f81f2b8e7af5.jpg)

<a id="paper-1034"></a>

## XSpecMesh: Quality-Preserving Auto-Regressive Mesh Generation Acceleration via Multi-Head Speculative Decoding


### Images

![08392882fdfacca78c78f0e2a8384c9bbfa25312ffbba1e92642a6988787db29.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/08392882fdfacca78c78f0e2a8384c9bbfa25312ffbba1e92642a6988787db29.jpg)

![4084ee0ff8773f7e806134f629de91ceecbbd602223afaa87343d507f9d28350.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/4084ee0ff8773f7e806134f629de91ceecbbd602223afaa87343d507f9d28350.jpg)

![521d6045c35f2e3ec3af5bfbc6efb52bffb3b3a7173e7755b1ee59a13576889a.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/521d6045c35f2e3ec3af5bfbc6efb52bffb3b3a7173e7755b1ee59a13576889a.jpg)

![8651faa3d38588d034b1e976f377fb502aad229f75f7b2002433b002fbff19a4.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/8651faa3d38588d034b1e976f377fb502aad229f75f7b2002433b002fbff19a4.jpg)

![88db4cebf338a55ed5229f5a7c62354c4a3e5ca2bd78443a280a7c9095e00817.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/88db4cebf338a55ed5229f5a7c62354c4a3e5ca2bd78443a280a7c9095e00817.jpg)

![9cf71e7c87a7278bd4b9759f251029680f7b60416cc66549f98d22979e96e34a.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/9cf71e7c87a7278bd4b9759f251029680f7b60416cc66549f98d22979e96e34a.jpg)

![d7705166e80bcb0a2f3ec7da92b590d97be291adb2d126d027e12431571a7e99.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/d7705166e80bcb0a2f3ec7da92b590d97be291adb2d126d027e12431571a7e99.jpg)

![edf11915d9dae635eb14bad4730baca9139207a15b6af0c5160a5929b548aaf3.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/edf11915d9dae635eb14bad4730baca9139207a15b6af0c5160a5929b548aaf3.jpg)

![f90c97a6491758a39b9497ce7ea4e3f3a34e55c04e42b0567a372391d80a4cce.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/images/f90c97a6491758a39b9497ce7ea4e3f3a34e55c04e42b0567a372391d80a4cce.jpg)

### Tables

![2015593035f9c81fe8933c29c0f0dc1f3702a6eead8f9c6a8719531cb16258ec.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/2015593035f9c81fe8933c29c0f0dc1f3702a6eead8f9c6a8719531cb16258ec.jpg)

![395e7b224f73057f7342fc4157e817a127d47e969ca9a8e9ca887d548035824c.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/395e7b224f73057f7342fc4157e817a127d47e969ca9a8e9ca887d548035824c.jpg)

![84e0c2fb6bce076a93d4b5a4a99110a669e3dd96b7375633316c9e0c91d9ef24.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/84e0c2fb6bce076a93d4b5a4a99110a669e3dd96b7375633316c9e0c91d9ef24.jpg)

![87facfad6b4153eb748b650432f1b2587b5465000440222b5972d5c1303b3674.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/87facfad6b4153eb748b650432f1b2587b5465000440222b5972d5c1303b3674.jpg)

![acc6e2edbe9895316c76e16e50b010648049eddaf406770f4b33af6a32879cf4.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/acc6e2edbe9895316c76e16e50b010648049eddaf406770f4b33af6a32879cf4.jpg)

![aedbac1596af50d82b65151e16cc0c2ee2c61ac7c720c085913503f3c6ed9c0f.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/aedbac1596af50d82b65151e16cc0c2ee2c61ac7c720c085913503f3c6ed9c0f.jpg)

![af4ef710170a635e9effa95d7310fa11ff62eea096b12d2af65726d0977cffa7.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/af4ef710170a635e9effa95d7310fa11ff62eea096b12d2af65726d0977cffa7.jpg)

![b6ab58ac006e47f80ff3f4306ee8684cff48881dd6c5b1414497d9fb73e56b40.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/b6ab58ac006e47f80ff3f4306ee8684cff48881dd6c5b1414497d9fb73e56b40.jpg)

![d4e366f7ab3429f600fbe81f3c53d917afa0a7592ca3b32edce3ab58e7a549ec.jpg](../icml_results/1052_XSpecMesh_%20Quality-Preserving%20Auto-Regressive%20Mesh%20Generation%20Acceleration%20via%20Multi-Head%20Speculativ/tables/d4e366f7ab3429f600fbe81f3c53d917afa0a7592ca3b32edce3ab58e7a549ec.jpg)

<a id="paper-1035"></a>

## Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting


### Images

![08615b09d46d2451a51cd4142e5bfe91dd817c15fa32590f4e0039eecc19092e.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/08615b09d46d2451a51cd4142e5bfe91dd817c15fa32590f4e0039eecc19092e.jpg)

![325eda50248b036ce3136adc4acadf18fcf3e7a875d5e172ea47a399d3a72fcd.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/325eda50248b036ce3136adc4acadf18fcf3e7a875d5e172ea47a399d3a72fcd.jpg)

![3bf03817e5224f1b64d748a80f764dce6f34d0e0866151c7a62726419dec638c.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/3bf03817e5224f1b64d748a80f764dce6f34d0e0866151c7a62726419dec638c.jpg)

![5b4626a61bd8b4c6c977f33ac374a544c5e3f91f7bcad6d6bd86eeba2da5af1a.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/5b4626a61bd8b4c6c977f33ac374a544c5e3f91f7bcad6d6bd86eeba2da5af1a.jpg)

![6e190ec8a3c22a4a03bcc07640d0cc044b7c16340820e1407f4e36f21333a6c1.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/6e190ec8a3c22a4a03bcc07640d0cc044b7c16340820e1407f4e36f21333a6c1.jpg)

![8f7b9a8b6f2ffa9ddd8356f2daa2eecb1b3728a84d364b7f6a29ee0db02d191d.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/8f7b9a8b6f2ffa9ddd8356f2daa2eecb1b3728a84d364b7f6a29ee0db02d191d.jpg)

![99c4780a8162d990838e0285304366f580afe93ec727ac3604dc290282ea6866.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/99c4780a8162d990838e0285304366f580afe93ec727ac3604dc290282ea6866.jpg)

![b33ce06cf5303dc0c49c01b9b163f4d0f4dcde99d37085aef66474ab10095844.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/b33ce06cf5303dc0c49c01b9b163f4d0f4dcde99d37085aef66474ab10095844.jpg)

![b69b936e168d0014845e9f439c1d706a842780ce33e971431a6c9fa17f258e1a.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/b69b936e168d0014845e9f439c1d706a842780ce33e971431a6c9fa17f258e1a.jpg)

![f4059398888d25574b7ad35d76a5f45bec0e956b6d93215c8d511d0e81c35c39.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/images/f4059398888d25574b7ad35d76a5f45bec0e956b6d93215c8d511d0e81c35c39.jpg)

### Tables

![4e1587f9c7a9f0b2742500cf916c6f1d0ae0aa07c62fc94230d3238517d23fd8.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/tables/4e1587f9c7a9f0b2742500cf916c6f1d0ae0aa07c62fc94230d3238517d23fd8.jpg)

![a0cb3ae1484867d81257456be787e4a22ce96f83382754b0b251fc83c500ddc5.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/tables/a0cb3ae1484867d81257456be787e4a22ce96f83382754b0b251fc83c500ddc5.jpg)

![ec152b4d3ff3c1e5f1bb310e044ba3dc711c8d2b0d48025fa0e945def4a16bc2.jpg](../icml_results/1053_Retaining%20by%20Doing_%20The%20Role%20of%20On-Policy%20Data%20in%20Mitigating%20Forgetting/tables/ec152b4d3ff3c1e5f1bb310e044ba3dc711c8d2b0d48025fa0e945def4a16bc2.jpg)

<a id="paper-1036"></a>

## Scam2Prompt: A Scalable Framework for Auditing Malicious Scam Endpoints in Production LLMs


### Images

![0a58e6c478b2f2cce855f14790231ef14efeabfc479d04aa63df2bc03a605611.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/0a58e6c478b2f2cce855f14790231ef14efeabfc479d04aa63df2bc03a605611.jpg)

![25aaf5ad226d0d039a814bd440d6870c3c81246c9acd09d1a6cfc0f0c7e5a6f7.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/25aaf5ad226d0d039a814bd440d6870c3c81246c9acd09d1a6cfc0f0c7e5a6f7.jpg)

![354dff05333e557acb1daf85158b6be6a339d60f2a1306d04408803a939675ab.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/354dff05333e557acb1daf85158b6be6a339d60f2a1306d04408803a939675ab.jpg)

![3a3cfe9a293b4e3982518dd56cdff181180573c49251f1eacc07701ae771deda.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/3a3cfe9a293b4e3982518dd56cdff181180573c49251f1eacc07701ae771deda.jpg)

![3f54f2d1157d024c35cb432794d789602921b5dfc05a9d91323c0daea99d1934.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/3f54f2d1157d024c35cb432794d789602921b5dfc05a9d91323c0daea99d1934.jpg)

![5f2ad0422516f9b3cedfccdd173a7d17891a9aa58ef08f1a3d8f069142bd2af3.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/5f2ad0422516f9b3cedfccdd173a7d17891a9aa58ef08f1a3d8f069142bd2af3.jpg)

![653d571b780efeac625bd5a3f3a000a202d8ba796101c61cfc5f8ab1250df793.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/653d571b780efeac625bd5a3f3a000a202d8ba796101c61cfc5f8ab1250df793.jpg)

![81b3d70bbf05b58661c828c0ffda74cec241e940dfa4b41c3d25a3586044be9f.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/81b3d70bbf05b58661c828c0ffda74cec241e940dfa4b41c3d25a3586044be9f.jpg)

![84623defa2fe57216ac207a8e630faff3c47337f6dfae302c8e455c7569ecaab.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/84623defa2fe57216ac207a8e630faff3c47337f6dfae302c8e455c7569ecaab.jpg)

![8552e53d99470957c510317193590510430181e45ac1e474f3beb2a7d123b5c2.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/8552e53d99470957c510317193590510430181e45ac1e474f3beb2a7d123b5c2.jpg)

![878de191a1bc936e40c00c86ea2446945159689ee88b6cb56596c1c707189aa5.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/878de191a1bc936e40c00c86ea2446945159689ee88b6cb56596c1c707189aa5.jpg)

![8e919317cd68d4012da3d31edfc9793be0f1382eda3203283ec094d94b847949.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/8e919317cd68d4012da3d31edfc9793be0f1382eda3203283ec094d94b847949.jpg)

![9d38c3b7230385e5be09f766638973fe105d767fa413921fd501f636e22d98fa.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/9d38c3b7230385e5be09f766638973fe105d767fa413921fd501f636e22d98fa.jpg)

![d9ec745c81ff11225a096f5866243efe9dd06639f434d416d646c5f37545df56.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/d9ec745c81ff11225a096f5866243efe9dd06639f434d416d646c5f37545df56.jpg)

![fd63d3852147c2d7c64ada5b89b729fc8add64b3d3a4780518c8076537f7f4b6.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/images/fd63d3852147c2d7c64ada5b89b729fc8add64b3d3a4780518c8076537f7f4b6.jpg)

### Tables

![0da0a3259660ce70284b3c1e44955db366a343293fcf1d6efee8edfecbfda760.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/0da0a3259660ce70284b3c1e44955db366a343293fcf1d6efee8edfecbfda760.jpg)

![18e7730765dc2800ddcfc305ecc1ddc328183d197a4f432f6283029ec3fd114f.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/18e7730765dc2800ddcfc305ecc1ddc328183d197a4f432f6283029ec3fd114f.jpg)

![257b388e6f873b71b1b5e075764389a1407dc17b0b567eb141be6041eaa44f70.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/257b388e6f873b71b1b5e075764389a1407dc17b0b567eb141be6041eaa44f70.jpg)

![39924963f3d18d16c42cbd49ec8e7d5caa2eb72704a4ddc3caf549fa94d19a37.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/39924963f3d18d16c42cbd49ec8e7d5caa2eb72704a4ddc3caf549fa94d19a37.jpg)

![4c09ac25d16604c5688ce524d646eabb9d4aba37f27557b2722f0789e2a78248.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/4c09ac25d16604c5688ce524d646eabb9d4aba37f27557b2722f0789e2a78248.jpg)

![7a4e1569fe7d87cc967ebe34cbdb9e31873d5d69f02180723c35fc836092500a.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/7a4e1569fe7d87cc967ebe34cbdb9e31873d5d69f02180723c35fc836092500a.jpg)

![86d551b08e0c77f8ebaf839be24d3ab04d9b3f8ffd35bb34bbd2fd599434f270.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/86d551b08e0c77f8ebaf839be24d3ab04d9b3f8ffd35bb34bbd2fd599434f270.jpg)

![a27539f8b3171bb5a822377ccde576a0c550f36d9ffe358ea55491f207cffe3b.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/a27539f8b3171bb5a822377ccde576a0c550f36d9ffe358ea55491f207cffe3b.jpg)

![c46dc406a3bcdbc6905429e29b12c9c4816bcc35d99c978b1569dc00addd1727.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/c46dc406a3bcdbc6905429e29b12c9c4816bcc35d99c978b1569dc00addd1727.jpg)

![ce661ea0896be5ff34c883c068fa3af231de729d43999ba4d070d27b3544d9d7.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/ce661ea0896be5ff34c883c068fa3af231de729d43999ba4d070d27b3544d9d7.jpg)

![ea38b71d3afdc3e18a6aa5077475961b812e9031b6e1c4372148158f6dcbec1b.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/ea38b71d3afdc3e18a6aa5077475961b812e9031b6e1c4372148158f6dcbec1b.jpg)

![f7a8219a1ca411b256e8738eeb3955609710e9d1d7d00018cdf119cd8419a28e.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/f7a8219a1ca411b256e8738eeb3955609710e9d1d7d00018cdf119cd8419a28e.jpg)

![f7b5c283d4ce8b9df91df8aa87eda6e311fe2d5a62f1623e999c74ff0498ddd8.jpg](../icml_results/1054_Scam2Prompt_%20A%20Scalable%20Framework%20for%20Auditing%20Malicious%20Scam%20Endpoints%20in%20Production%20LLMs/tables/f7b5c283d4ce8b9df91df8aa87eda6e311fe2d5a62f1623e999c74ff0498ddd8.jpg)

<a id="paper-1037"></a>

## AudioChat: Unified Audio Storytelling, Editing, and Understanding with Transfusion Forcing


### Images

![037bd042647a3c3a4b25ea223c136dd1e6ee7a4c2f44757a43e3ada1f424a4d2.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/037bd042647a3c3a4b25ea223c136dd1e6ee7a4c2f44757a43e3ada1f424a4d2.jpg)

![0eea110b36723c5bcf397c37600591ff368e6baaaaf8eb427c545feb85a170a0.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/0eea110b36723c5bcf397c37600591ff368e6baaaaf8eb427c545feb85a170a0.jpg)

![1334ae7a725cd4ea8129156451a98ae6faf0ee1f145db3932c36a0c9e7226603.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/1334ae7a725cd4ea8129156451a98ae6faf0ee1f145db3932c36a0c9e7226603.jpg)

![27f7a3b0f6ec0938e6e70e91668e35c35d12dccecef235f425ba0c9c0ea17f9f.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/27f7a3b0f6ec0938e6e70e91668e35c35d12dccecef235f425ba0c9c0ea17f9f.jpg)

![6cf5388c6fbc2c5d97f81ca9f0a342e488d8dbcff3e214a6cc236a702dbe08dd.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/6cf5388c6fbc2c5d97f81ca9f0a342e488d8dbcff3e214a6cc236a702dbe08dd.jpg)

![bfe728515e0822ba2830fa02e1dc8d30d823e66fc4c815efd45f52c27fbd2c63.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/bfe728515e0822ba2830fa02e1dc8d30d823e66fc4c815efd45f52c27fbd2c63.jpg)

![c76475f66ce93c834e9a50e565cb61146155bb26211bc1e5d3ee06174ed9fdce.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/images/c76475f66ce93c834e9a50e565cb61146155bb26211bc1e5d3ee06174ed9fdce.jpg)

### Tables

![11ac3bb489d8093303512e0474eb7281532b695b8de52b9f235db7183e82dab7.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/11ac3bb489d8093303512e0474eb7281532b695b8de52b9f235db7183e82dab7.jpg)

![2f339a1aecfe2c262fa0ad87c478d76d115c5526646845256bf612cb6f73f321.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/2f339a1aecfe2c262fa0ad87c478d76d115c5526646845256bf612cb6f73f321.jpg)

![305ed896a708c7bdedf5019ef7ef8062746861ea9222028a653bff2c057db706.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/305ed896a708c7bdedf5019ef7ef8062746861ea9222028a653bff2c057db706.jpg)

![3a02620a51eeb3fb456da870612a34b4b1a304fdcc2ab551b15dc28f963b1ef5.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/3a02620a51eeb3fb456da870612a34b4b1a304fdcc2ab551b15dc28f963b1ef5.jpg)

![4174dead4f4f02e4779c93493cf85748ac6ecd0a137646fd6bce8a0afeab0543.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/4174dead4f4f02e4779c93493cf85748ac6ecd0a137646fd6bce8a0afeab0543.jpg)

![749adc078c743a2956e68aaea218dccd7b5ede0a182baa97395f40a7f3b3a41c.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/749adc078c743a2956e68aaea218dccd7b5ede0a182baa97395f40a7f3b3a41c.jpg)

![7bb809fa13b74ebc4d05b0d7e69c195bcef55d64f1c818b55bcd920316d7f757.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/7bb809fa13b74ebc4d05b0d7e69c195bcef55d64f1c818b55bcd920316d7f757.jpg)

![ef4c7871869ef07430c0e79b7bf4d31a5f2af472ee96afd32d78b0b551d859df.jpg](../icml_results/1055_AudioChat_%20Unified%20Audio%20Storytelling%2C%20Editing%2C%20and%20Understanding%20with%20Transfusion%20Forcing/tables/ef4c7871869ef07430c0e79b7bf4d31a5f2af472ee96afd32d78b0b551d859df.jpg)

<a id="paper-1038"></a>

## Beyond External Monitors: Enhancing Transparency of Large Language Models for Easier Monitoring


### Images

![0002742f493f19543ae47e795670b3dddb3cd0b8cae41e1dbadbe78e756c6937.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/0002742f493f19543ae47e795670b3dddb3cd0b8cae41e1dbadbe78e756c6937.jpg)

![22bcf541a5926571144f8d4cab8572ed0558b4a4e20b998b422a34616d2c41d4.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/22bcf541a5926571144f8d4cab8572ed0558b4a4e20b998b422a34616d2c41d4.jpg)

![2abf9936ccb2380f61e7a97504240f5302d722336c0cf033560c71ad58286568.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/2abf9936ccb2380f61e7a97504240f5302d722336c0cf033560c71ad58286568.jpg)

![2c94aa6025c81f93aadd895dbbee8c83a0747c03a96ce756aa0f376c8d9dd089.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/2c94aa6025c81f93aadd895dbbee8c83a0747c03a96ce756aa0f376c8d9dd089.jpg)

![47554cdeee1083a92510b04c9efea70d275bf36e2e3761b22b3d24184484ef05.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/47554cdeee1083a92510b04c9efea70d275bf36e2e3761b22b3d24184484ef05.jpg)

![679cdc6127e2b42b85e0c16e510d87f427feb6f5ce2fd93c16b119bd4bc08f9e.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/679cdc6127e2b42b85e0c16e510d87f427feb6f5ce2fd93c16b119bd4bc08f9e.jpg)

![71c1ebef8fc50ae20c8d3193e9aac57b075bf91aaf45644268fee211dc2c4635.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/71c1ebef8fc50ae20c8d3193e9aac57b075bf91aaf45644268fee211dc2c4635.jpg)

![73bd11c3a97d4c239fcb2818131a60fefb4c453e0f5e241d7724eb3e1be37ec5.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/73bd11c3a97d4c239fcb2818131a60fefb4c453e0f5e241d7724eb3e1be37ec5.jpg)

![dd02e9e789408b5792f1081970d5cd9eef0d039153a3e8a70d7efa5ff3328714.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/images/dd02e9e789408b5792f1081970d5cd9eef0d039153a3e8a70d7efa5ff3328714.jpg)

### Tables

![0464e0acf739290ec4526bdeb2ccf914990d536fda6f9cfa89bbfd84bf0f459c.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/0464e0acf739290ec4526bdeb2ccf914990d536fda6f9cfa89bbfd84bf0f459c.jpg)

![05cde4a4f558975668c1c12da27e6a20a78a52792f9d7ee9034cf98d4b5a1654.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/05cde4a4f558975668c1c12da27e6a20a78a52792f9d7ee9034cf98d4b5a1654.jpg)

![0a0fa5f39c93de3584558c9f97f2f9f519d6d97099a071eade55da71ff0ac995.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/0a0fa5f39c93de3584558c9f97f2f9f519d6d97099a071eade55da71ff0ac995.jpg)

![1f934ecfba169f47bb818ba75fac1090fe0ed373b55543472739de871441eb34.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/1f934ecfba169f47bb818ba75fac1090fe0ed373b55543472739de871441eb34.jpg)

![21604c12918caf37a444b6697e717b41bf6aa9f1b294cd0cd2d9702ca0803d77.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/21604c12918caf37a444b6697e717b41bf6aa9f1b294cd0cd2d9702ca0803d77.jpg)

![243e167a1f4cc85256c188b94252616c2f869f217bf58a100c5a8cc8aec08d5d.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/243e167a1f4cc85256c188b94252616c2f869f217bf58a100c5a8cc8aec08d5d.jpg)

![2b1047f675f69c811dfcc4a87c59c084de0a480f54f72b8b14e03b8993b7e535.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/2b1047f675f69c811dfcc4a87c59c084de0a480f54f72b8b14e03b8993b7e535.jpg)

![8878ad08dd6a7356932680195ecac6db55a275bd937ff6449d59f429795b8a25.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/8878ad08dd6a7356932680195ecac6db55a275bd937ff6449d59f429795b8a25.jpg)

![90dce92016b12a89518625a6ffa775127836c4b29277e89e1eb42e5dea208752.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/90dce92016b12a89518625a6ffa775127836c4b29277e89e1eb42e5dea208752.jpg)

![a2290da6d86670a9091453181a02cd7505fa9c81dc8b92f83f6932547c8f6497.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/a2290da6d86670a9091453181a02cd7505fa9c81dc8b92f83f6932547c8f6497.jpg)

![b48f6bff72836e64d40e928856053e34a6cc8b9093043a70ff525c181d1f776a.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/b48f6bff72836e64d40e928856053e34a6cc8b9093043a70ff525c181d1f776a.jpg)

![ba6204e140bc48b7b47ad15eba8544b83483b3b844f5559715ed360b1583c4ac.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/ba6204e140bc48b7b47ad15eba8544b83483b3b844f5559715ed360b1583c4ac.jpg)

![bc13706f5e2c4e92e2a25a88e62043769877dab62fc7bae68c58ccb43208edec.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/bc13706f5e2c4e92e2a25a88e62043769877dab62fc7bae68c58ccb43208edec.jpg)

![c9c8c36fd8bcce536f7395531b85d6229d20cc1b1a6bf61918e7ab03eec14691.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/c9c8c36fd8bcce536f7395531b85d6229d20cc1b1a6bf61918e7ab03eec14691.jpg)

![cccabacdfd6ef4a35866f6bf4a40f523b0cf93e08f774830fc4d4ddd0bfdee5e.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/cccabacdfd6ef4a35866f6bf4a40f523b0cf93e08f774830fc4d4ddd0bfdee5e.jpg)

![d50ceae0548f22ef0c620ab9a005532fdbbd7c9f493c2b6ecf68be065a2f906a.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/d50ceae0548f22ef0c620ab9a005532fdbbd7c9f493c2b6ecf68be065a2f906a.jpg)

![d54e8d1e28a1647d378c161a88c10efe7b310759c648c221352e543dba8aa78c.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/d54e8d1e28a1647d378c161a88c10efe7b310759c648c221352e543dba8aa78c.jpg)

![edf9bccf3839d07c1cd7496d1de734edab09561079892e4be05fc01a06da207b.jpg](../icml_results/1056_Beyond%20External%20Monitors_%20Enhancing%20Transparency%20of%20Large%20Language%20Models%20for%20Easier%20Monitoring/tables/edf9bccf3839d07c1cd7496d1de734edab09561079892e4be05fc01a06da207b.jpg)

<a id="paper-1039"></a>

## Adversarially Robust Control of Conditional Value-at-Risk via Rockafellar-Uryasev Conformal Inference


### Images

![12d7783c428bda4cebfd9592f67380f4cda203c08f6551fe7c295685e772e857.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/12d7783c428bda4cebfd9592f67380f4cda203c08f6551fe7c295685e772e857.jpg)

![27e95b8670762b3885b25a2a3156b5a30d3fcf187eae069b9e3c9ad84980ae3d.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/27e95b8670762b3885b25a2a3156b5a30d3fcf187eae069b9e3c9ad84980ae3d.jpg)

![2b0f38287c460d541f41e14d3fe23365a7543f0f2ddfdda0e30d4ea02dafdc50.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/2b0f38287c460d541f41e14d3fe23365a7543f0f2ddfdda0e30d4ea02dafdc50.jpg)

![38556c6d2a75f20ec038b339ddc7eebc775e12c10258d56fa996044b5c79a2bb.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/38556c6d2a75f20ec038b339ddc7eebc775e12c10258d56fa996044b5c79a2bb.jpg)

![6a713cf86f36b19ffa07993ea054b332e7dd1a8ef13a7e50c93fb2fbff3c197b.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/6a713cf86f36b19ffa07993ea054b332e7dd1a8ef13a7e50c93fb2fbff3c197b.jpg)

![6c30029a43514037eeb7750a6ea1f2c52a7b8326891745bd155eb70ac9dd754c.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/6c30029a43514037eeb7750a6ea1f2c52a7b8326891745bd155eb70ac9dd754c.jpg)

![8c0e094d227e4824b544731d971d834cbce1122707c766ad7c607f69132f0061.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/8c0e094d227e4824b544731d971d834cbce1122707c766ad7c607f69132f0061.jpg)

![acb2022785f8890c5d7413952b0008535750c8d6210ebfb3cfa87eb965611db4.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/acb2022785f8890c5d7413952b0008535750c8d6210ebfb3cfa87eb965611db4.jpg)

![c2866b019374bee4c80d9356fe525e2de98f3cef3864c91801d2a0eca56662f6.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/c2866b019374bee4c80d9356fe525e2de98f3cef3864c91801d2a0eca56662f6.jpg)

![f20339d87dd7a384e15b51f6327c0f7c11ed2f93867dda07b52358f81ab0abde.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/f20339d87dd7a384e15b51f6327c0f7c11ed2f93867dda07b52358f81ab0abde.jpg)

![f32004da0c2050950a1f94b08e51cc08d5b7ec44ff4beeb4b8079d739ab971c4.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/f32004da0c2050950a1f94b08e51cc08d5b7ec44ff4beeb4b8079d739ab971c4.jpg)

![ffc10414d092cea0f458d4a106bea4ea75ef0b2e4085ffb5b91ae73b729fce02.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/images/ffc10414d092cea0f458d4a106bea4ea75ef0b2e4085ffb5b91ae73b729fce02.jpg)

### Tables

![eb3a10854f383a5620a6c05c2a5048618bd42508f90a909225a5213e3d2bad9c.jpg](../icml_results/1057_Adversarially%20Robust%20Control%20of%20Conditional%20Value-at-Risk%20via%20Rockafellar-Uryasev%20Conformal%20Inferenc/tables/eb3a10854f383a5620a6c05c2a5048618bd42508f90a909225a5213e3d2bad9c.jpg)

<a id="paper-1040"></a>

## SMILE: Extended Deep Submodular Function-Based Instruction and In-context Learning Demonstration Selection


### Images

![1429525bd686e4f2ced112fa8a046a3d655d7580ab7a593670f79e4778025a89.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/images/1429525bd686e4f2ced112fa8a046a3d655d7580ab7a593670f79e4778025a89.jpg)

![adbfa6f7e5e3907a40d58219709c83b09a3c0899a2d0817d3d58b7f0a068fa02.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/images/adbfa6f7e5e3907a40d58219709c83b09a3c0899a2d0817d3d58b7f0a068fa02.jpg)

![c4126d45ed7bb841e6a52b0aa875524f0bf4832d3c57119108168f0265cb6bf5.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/images/c4126d45ed7bb841e6a52b0aa875524f0bf4832d3c57119108168f0265cb6bf5.jpg)

![e3c4346d0c6f7e401089177eac1711295bd122e238d1d88a6fcd318495236482.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/images/e3c4346d0c6f7e401089177eac1711295bd122e238d1d88a6fcd318495236482.jpg)

### Tables

![1127fbe2e120b0972eb62afae2a4c0cc2ee476a1b406151e2c85e4759d573eeb.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/1127fbe2e120b0972eb62afae2a4c0cc2ee476a1b406151e2c85e4759d573eeb.jpg)

![3a36a6dc363494295180957640412308e734ba0a82b0e95b64e55f89cbf42483.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/3a36a6dc363494295180957640412308e734ba0a82b0e95b64e55f89cbf42483.jpg)

![3b81b9b794b7b95e436cfeb547bf3afb78ec479c3a7f3c0794c8ba1e14b6336f.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/3b81b9b794b7b95e436cfeb547bf3afb78ec479c3a7f3c0794c8ba1e14b6336f.jpg)

![5038a0d18401e753002ba36e515a436d50f35b656ce5c5f29f51aff5577bf000.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/5038a0d18401e753002ba36e515a436d50f35b656ce5c5f29f51aff5577bf000.jpg)

![5958e5c097dddba67650335420793729de3a3a459caab3ab9a47ed104c31422b.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/5958e5c097dddba67650335420793729de3a3a459caab3ab9a47ed104c31422b.jpg)

![63ea62a60d3a4346885d2a71c291654899cb620543da5d51747a0ad94ba49461.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/63ea62a60d3a4346885d2a71c291654899cb620543da5d51747a0ad94ba49461.jpg)

![6715a57d8a840b10cf1a1d1c7d5f6cc997d72cfd1ff0199ff1ee5963ea83083b.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/6715a57d8a840b10cf1a1d1c7d5f6cc997d72cfd1ff0199ff1ee5963ea83083b.jpg)

![727f5ff8882d14079cc2cb0953250ef9d9a1cd92a2c0a7ff5714e746150bf9f9.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/727f5ff8882d14079cc2cb0953250ef9d9a1cd92a2c0a7ff5714e746150bf9f9.jpg)

![c479f81642d33732fdf8f38efe7c86f5b01389f13fd5f55cf2897f89418646c3.jpg](../icml_results/1058_SMILE_%20Extended%20Deep%20Submodular%20Function-Based%20Instruction%20and%20In-context%20Learning%20Demonstration%20Sel/tables/c479f81642d33732fdf8f38efe7c86f5b01389f13fd5f55cf2897f89418646c3.jpg)

<a id="paper-1041"></a>

## Autoregression with Self-Token Prediction


### Images

![069eadc0ab52bfa79e6c12966aada47065c952b61f65482cba05051e678b198f.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/069eadc0ab52bfa79e6c12966aada47065c952b61f65482cba05051e678b198f.jpg)

![0b82573e25f0587291af9842f94ed1c9f1a0d095e67e38ec0c6fd82302faf981.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/0b82573e25f0587291af9842f94ed1c9f1a0d095e67e38ec0c6fd82302faf981.jpg)

![13e5af7dbae34d8162e364b8676e1c5ab1d529314e4cfff2039ad373a2130d60.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/13e5af7dbae34d8162e364b8676e1c5ab1d529314e4cfff2039ad373a2130d60.jpg)

![22efe26f07206acc9f1043df6e53ce8972af917871c6c9a41ab0d12b012ddb03.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/22efe26f07206acc9f1043df6e53ce8972af917871c6c9a41ab0d12b012ddb03.jpg)

![2697b97e58a51389642f0b1daeae50397fc201d0098a22214a514b810e45a62a.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/2697b97e58a51389642f0b1daeae50397fc201d0098a22214a514b810e45a62a.jpg)

![386a5dacec46cd18a330140d19f0fcedb6fd32220299a9cda12c20e888394e87.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/386a5dacec46cd18a330140d19f0fcedb6fd32220299a9cda12c20e888394e87.jpg)

![390b0ba08e6b3d3afc1ecefe3b40194efdf7c302b8fb875ac11cb1dfa022cc54.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/390b0ba08e6b3d3afc1ecefe3b40194efdf7c302b8fb875ac11cb1dfa022cc54.jpg)

![4f02264c4a93118fa50ff46328aee9e3588089942117499ea9de1701ccc7edb8.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/4f02264c4a93118fa50ff46328aee9e3588089942117499ea9de1701ccc7edb8.jpg)

![580062c2884d2aa24eff1f3ca29bd7dd5a0034083017e8c2b0b421d7e9a4170c.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/580062c2884d2aa24eff1f3ca29bd7dd5a0034083017e8c2b0b421d7e9a4170c.jpg)

![64826819ffd25148edc34496b0c25bc55cf51142ecad3de920b4eb15cdd64514.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/64826819ffd25148edc34496b0c25bc55cf51142ecad3de920b4eb15cdd64514.jpg)

![7bbade5abf22a6d7221a60e374029c5e8a418b678a1d01395b2222134c9984d6.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/7bbade5abf22a6d7221a60e374029c5e8a418b678a1d01395b2222134c9984d6.jpg)

![7d1188e98cd86dc5c5e900c63cd15aaf6db23da9cf2f004c4ffc4fc7bde1f3fa.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/7d1188e98cd86dc5c5e900c63cd15aaf6db23da9cf2f004c4ffc4fc7bde1f3fa.jpg)

![8c87569ab0718a01109e029bb9f038bca33ac1a0ae5334ca53853718d4388d13.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/8c87569ab0718a01109e029bb9f038bca33ac1a0ae5334ca53853718d4388d13.jpg)

![90f10a75cccf2a9fc95ebe513e0977be1ed40df04dc641ba6f4cd9e9d3c0bbbb.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/90f10a75cccf2a9fc95ebe513e0977be1ed40df04dc641ba6f4cd9e9d3c0bbbb.jpg)

![a250e775bced65607bfcc530d9b77a6d438ead121e575609fcb76775dc0177a9.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/a250e775bced65607bfcc530d9b77a6d438ead121e575609fcb76775dc0177a9.jpg)

![adfb3e6b20868cc85ccf49cca744d20e549a6782c616c2f7ea49543900e844b6.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/adfb3e6b20868cc85ccf49cca744d20e549a6782c616c2f7ea49543900e844b6.jpg)

![b6f4e162e5adecdc61e81b3ce8e506371f2f3eaeab58b86d1c2341cbf96612f9.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/b6f4e162e5adecdc61e81b3ce8e506371f2f3eaeab58b86d1c2341cbf96612f9.jpg)

![bd00f286a1eef6722d6921dd407f7f5ee11929ca0af5d687c51d5c3aa58d8fee.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/bd00f286a1eef6722d6921dd407f7f5ee11929ca0af5d687c51d5c3aa58d8fee.jpg)

![c2192daef52b302445ebd0529a42891ba414dbd3e3f520785de87c9cc286d2dd.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/c2192daef52b302445ebd0529a42891ba414dbd3e3f520785de87c9cc286d2dd.jpg)

![ca2a2cd09c81444088ea38b2e9c2c27376efa04115de186408a0f720c20629f1.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/ca2a2cd09c81444088ea38b2e9c2c27376efa04115de186408a0f720c20629f1.jpg)

![d662f00e6d62658d32e1e1be5ff41ba51d0ec631a334ebff3af8aad62f485e76.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/d662f00e6d62658d32e1e1be5ff41ba51d0ec631a334ebff3af8aad62f485e76.jpg)

![d9219dc7120c69081aa425dd80b4c04bb2232ed9d866973c32ef2e0fc569e81a.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/d9219dc7120c69081aa425dd80b4c04bb2232ed9d866973c32ef2e0fc569e81a.jpg)

![e93f8dcda51e8094c2b5b36fb5cf3538af7eda025c7fe41480f791c8b7e4a3b6.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/e93f8dcda51e8094c2b5b36fb5cf3538af7eda025c7fe41480f791c8b7e4a3b6.jpg)

![ff8fe1456f7e111a5635fd41662424e31d41ca757e9ae2fb63431434b825a1ab.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/images/ff8fe1456f7e111a5635fd41662424e31d41ca757e9ae2fb63431434b825a1ab.jpg)

### Tables

![01fa4bee0b5f56eef1f10dc4e85222892060286b8b4d33547bbbb1fdc54d8177.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/tables/01fa4bee0b5f56eef1f10dc4e85222892060286b8b4d33547bbbb1fdc54d8177.jpg)

![2b3edc5592d7d9ed62dc87c3a53ce35cd796d435e7b82fda1cfff16d90d51416.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/tables/2b3edc5592d7d9ed62dc87c3a53ce35cd796d435e7b82fda1cfff16d90d51416.jpg)

![9d1bba28ec156b9b1244a3f25dfb99755587ccb4f721269d64aac6da54f76bd0.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/tables/9d1bba28ec156b9b1244a3f25dfb99755587ccb4f721269d64aac6da54f76bd0.jpg)

![e0b582d7da5455aa14c8b57f9917ac372051cfce612047363d1f6921f249fad2.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/tables/e0b582d7da5455aa14c8b57f9917ac372051cfce612047363d1f6921f249fad2.jpg)

![e198255358bc3ec0b954ec0577cc8755019f717fb0480b8a129b40eafd205baf.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/tables/e198255358bc3ec0b954ec0577cc8755019f717fb0480b8a129b40eafd205baf.jpg)

![f483783e95a66d202ae84d8aade86c270b479730b84c3894fbfdbbafeb23c61a.jpg](../icml_results/1059_Autoregression%20with%20Self-Token%20Prediction/tables/f483783e95a66d202ae84d8aade86c270b479730b84c3894fbfdbbafeb23c61a.jpg)

<a id="paper-1042"></a>

## SPR: A Structured Prompt Refinement Network for Modality Missing


### Images

![0fc260513304cdb9ef3e5fb1d0ceaf47f7e9bbd185b10f169d072970cb82899d.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/0fc260513304cdb9ef3e5fb1d0ceaf47f7e9bbd185b10f169d072970cb82899d.jpg)

![1aeb6ec543fe12d3cf7ca0d9458239c5798a16c3c1d7ed1b59e209cacbf7edca.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/1aeb6ec543fe12d3cf7ca0d9458239c5798a16c3c1d7ed1b59e209cacbf7edca.jpg)

![1b411dd710c4501fe635229384a721f61770eaa739227bf52d89b6c21591126a.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/1b411dd710c4501fe635229384a721f61770eaa739227bf52d89b6c21591126a.jpg)

![2d893be709160ac95af769277956d89b541ab2509253a3684cecae1fff4fbcda.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/2d893be709160ac95af769277956d89b541ab2509253a3684cecae1fff4fbcda.jpg)

![48d427e60df07d6ddac870f4acd62205d5fc49da97ea73f60b172b38ea5276a7.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/48d427e60df07d6ddac870f4acd62205d5fc49da97ea73f60b172b38ea5276a7.jpg)

![54b111c1bd34cd8e4dede8efc7e37b7864c0f8b558ded76327ed550cc8648ad3.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/54b111c1bd34cd8e4dede8efc7e37b7864c0f8b558ded76327ed550cc8648ad3.jpg)

![a76901f2165af1c68897275924b3c867963859da53146ecf5600cf53c1193e1d.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/a76901f2165af1c68897275924b3c867963859da53146ecf5600cf53c1193e1d.jpg)

![b9b55952a58204ae98c007779c9812ee1ded45e21a25ec057c018ec48ebcf47d.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/b9b55952a58204ae98c007779c9812ee1ded45e21a25ec057c018ec48ebcf47d.jpg)

![bdad9c8ecff7aadbd1019fd4c1f494fd5aed97300539ff81dfea8947d3640b4f.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/bdad9c8ecff7aadbd1019fd4c1f494fd5aed97300539ff81dfea8947d3640b4f.jpg)

![c3aa719b8adc0dcff38e726fc63a9f2295e55b0c18da093fa227a64b18326920.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/c3aa719b8adc0dcff38e726fc63a9f2295e55b0c18da093fa227a64b18326920.jpg)

![c7306f0a8175e7cf670c5363e5edde1d504b35ad07c33b9ad123d8d9c8dddf23.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/c7306f0a8175e7cf670c5363e5edde1d504b35ad07c33b9ad123d8d9c8dddf23.jpg)

![cb206cd6408ec11e0e9d3471b4d2ff38b80758f3d713445bb34b89d52d678913.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/cb206cd6408ec11e0e9d3471b4d2ff38b80758f3d713445bb34b89d52d678913.jpg)

![ccff08710ae61b63c669d1bde60426da61c76427cfffb6273373cbbaa0158c5a.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/images/ccff08710ae61b63c669d1bde60426da61c76427cfffb6273373cbbaa0158c5a.jpg)

### Tables

![58e4e3a27684f0a093a882a4d532c3e6ce04e1903f79a3969e1df620dc9e9e06.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/tables/58e4e3a27684f0a093a882a4d532c3e6ce04e1903f79a3969e1df620dc9e9e06.jpg)

![e587e8c591b794d94329a8e02909fa1eb772ba68f172b8dc370e69f6006862aa.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/tables/e587e8c591b794d94329a8e02909fa1eb772ba68f172b8dc370e69f6006862aa.jpg)

![f84ec933b29a0cb71d2754459328398fb163a6cdd1d030bdfdffe682046992d3.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/tables/f84ec933b29a0cb71d2754459328398fb163a6cdd1d030bdfdffe682046992d3.jpg)

![fdafcbf8395eccf7d4aa0b0423c93c69314e5a6a33c5763a52c8b1da762420da.jpg](../icml_results/1060_SPR_%20A%20Structured%20Prompt%20Refinement%20Network%20for%20Modality%20Missing/tables/fdafcbf8395eccf7d4aa0b0423c93c69314e5a6a33c5763a52c8b1da762420da.jpg)

<a id="paper-1043"></a>

## Beyond Logits: Coherent Hallucination Mitigation via Attention Contrastive Decoding


### Images

![2aa42b92133d832d96c6e1db3febff909972e2d07c264378f86e95bb49845c69.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/2aa42b92133d832d96c6e1db3febff909972e2d07c264378f86e95bb49845c69.jpg)

![3027e4b0754e078b757791d9e050250253b7ca0673bb5e0883cfba478a883e43.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/3027e4b0754e078b757791d9e050250253b7ca0673bb5e0883cfba478a883e43.jpg)

![42641bee2d8d2983b8ffd807f99015f961c890e2fde2f089a3cdd34f7c5b13a3.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/42641bee2d8d2983b8ffd807f99015f961c890e2fde2f089a3cdd34f7c5b13a3.jpg)

![5a069e412ff106548a965fe0156313c9ef9c003aaf66bf4686580b5747cb52cc.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/5a069e412ff106548a965fe0156313c9ef9c003aaf66bf4686580b5747cb52cc.jpg)

![610035dc4280a4f4d7c616854585a08446d62b6b15f30d0a9086802407c3a653.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/610035dc4280a4f4d7c616854585a08446d62b6b15f30d0a9086802407c3a653.jpg)

![6380d857ad0f34f56629c7724fde4bfd415eb900a21bc66ed80592b301105ea3.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/6380d857ad0f34f56629c7724fde4bfd415eb900a21bc66ed80592b301105ea3.jpg)

![66ce87b217f44379733ca4212a27db70774cc800a159d448ba5cc0aa074356af.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/66ce87b217f44379733ca4212a27db70774cc800a159d448ba5cc0aa074356af.jpg)

![7aa30cbc7b18cd1852d71db4b5f1ba32b229be0ce45093a383a0d446bcdc88d9.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/7aa30cbc7b18cd1852d71db4b5f1ba32b229be0ce45093a383a0d446bcdc88d9.jpg)

![827c59081c02fc82514d758827509dc8c278b839ea29d5c10c8aeb35a92e4739.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/827c59081c02fc82514d758827509dc8c278b839ea29d5c10c8aeb35a92e4739.jpg)

![cb6e0fef5a6e12de87f56a60f985353f95319002b66708821bedd2cdb533e523.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/cb6e0fef5a6e12de87f56a60f985353f95319002b66708821bedd2cdb533e523.jpg)

![d215e3c48dbf8fea9a41f1001b70fe7293560e407644fd3bbc9cb8d0f12ac568.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/d215e3c48dbf8fea9a41f1001b70fe7293560e407644fd3bbc9cb8d0f12ac568.jpg)

![ddc2767e1f155a728942c81da2912a388c4437a0f31d50e2a8871d723d43fd32.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/images/ddc2767e1f155a728942c81da2912a388c4437a0f31d50e2a8871d723d43fd32.jpg)

### Tables

![06d0f8d3d50e5178bc57931dd09e11a2f8f612e2d2415536bb87170cf94c0e7b.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/06d0f8d3d50e5178bc57931dd09e11a2f8f612e2d2415536bb87170cf94c0e7b.jpg)

![06ea229b72da7149262c20e0525f41da78558c0fea6582652573db0f5f6bd812.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/06ea229b72da7149262c20e0525f41da78558c0fea6582652573db0f5f6bd812.jpg)

![1947cad8d7ee0ad0ed2927a6eb5368bc4b404081a32b8b8784d556738da2a01e.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/1947cad8d7ee0ad0ed2927a6eb5368bc4b404081a32b8b8784d556738da2a01e.jpg)

![1d09175c238f91b803aa2cc6e132b3647474de00e7f1c997dde009ee8c8534fc.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/1d09175c238f91b803aa2cc6e132b3647474de00e7f1c997dde009ee8c8534fc.jpg)

![2067ecabcfb93f4e999183937374887c3d7ae9c641c2fa7773ce39fbea72eaa6.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/2067ecabcfb93f4e999183937374887c3d7ae9c641c2fa7773ce39fbea72eaa6.jpg)

![3c49f9ec33ba6bf65a263e5b1c9ece8703d1208b080db48b6118f0fb612e544c.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/3c49f9ec33ba6bf65a263e5b1c9ece8703d1208b080db48b6118f0fb612e544c.jpg)

![53fb65424dafb6acb83ceb85f081413b0677d5edbe5959ca693d13b29d0190b2.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/53fb65424dafb6acb83ceb85f081413b0677d5edbe5959ca693d13b29d0190b2.jpg)

![6e3b22d9840fe8e4e182e944327f9b55a75b3d47f904702d27f4373659c66ec7.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/6e3b22d9840fe8e4e182e944327f9b55a75b3d47f904702d27f4373659c66ec7.jpg)

![83d4ffc13a9ff4383711e2ea0207a4bb78937007bbfb4c80e8d4707e3e389e2c.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/83d4ffc13a9ff4383711e2ea0207a4bb78937007bbfb4c80e8d4707e3e389e2c.jpg)

![9f4e8b552382638a49581e9e01a8b60b5e68aaca19e417413d9403209e140396.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/9f4e8b552382638a49581e9e01a8b60b5e68aaca19e417413d9403209e140396.jpg)

![ba67e1dba0e4e3b5233a51fedd7b7c736eb2affe3f9ceb4e7a5e4e7eab74baec.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/ba67e1dba0e4e3b5233a51fedd7b7c736eb2affe3f9ceb4e7a5e4e7eab74baec.jpg)

![c59dfcd2f76de15d49f8c667ea070b74ffa1fc03bddf10df62c2746664e4d3dc.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/c59dfcd2f76de15d49f8c667ea070b74ffa1fc03bddf10df62c2746664e4d3dc.jpg)

![ebff7b413105f056cab555fbb232f141a3f97c794c8a939b4ae46bea331cd5bd.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/ebff7b413105f056cab555fbb232f141a3f97c794c8a939b4ae46bea331cd5bd.jpg)

![ec17412cbcf51e17dc560aaacddf638f4b0d250c8229bc9f19b1a395c0448e91.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/ec17412cbcf51e17dc560aaacddf638f4b0d250c8229bc9f19b1a395c0448e91.jpg)

![ed2f7e48c66865b20b896bfdd8ed0bf5ae37536d111238748d9a8057c541f03a.jpg](../icml_results/1061_Beyond%20Logits_%20Coherent%20Hallucination%20Mitigation%20via%20Attention%20Contrastive%20Decoding/tables/ed2f7e48c66865b20b896bfdd8ed0bf5ae37536d111238748d9a8057c541f03a.jpg)

<a id="paper-1044"></a>

## Noise as a Natural Regularizer in Markov Decision Processes: Connecting Environmental Stochasticity and Policy Simplicity


### Images

![0262012dff2e0c957445c1b6f5f28f173160b61e4b68c17f498794acad6cb73d.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/0262012dff2e0c957445c1b6f5f28f173160b61e4b68c17f498794acad6cb73d.jpg)

![105b4ccc651fa98a33a85a598a1e42d1ea5d2a3ed9e4441f1b538b5c593c7a67.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/105b4ccc651fa98a33a85a598a1e42d1ea5d2a3ed9e4441f1b538b5c593c7a67.jpg)

![2a092199a8e265ba8f16b24db08378e97fe53853c247a298d7a8a069ace814b5.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/2a092199a8e265ba8f16b24db08378e97fe53853c247a298d7a8a069ace814b5.jpg)

![3e3bf5939c779d814bcfbf5ee1a6efa2974965b6bfab80dbd7a32a3578ff8450.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/3e3bf5939c779d814bcfbf5ee1a6efa2974965b6bfab80dbd7a32a3578ff8450.jpg)

![564d1694fd55316f6a5a04b1d81419d1be07868080c4e6f0c80d1c8a9a98b1a9.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/564d1694fd55316f6a5a04b1d81419d1be07868080c4e6f0c80d1c8a9a98b1a9.jpg)

![6affd4cfa2ade1f92c5017d5d59e4b9e5e48d6c7b56fab130316096d54b033eb.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/6affd4cfa2ade1f92c5017d5d59e4b9e5e48d6c7b56fab130316096d54b033eb.jpg)

![734120af2ad82465c1ba9a723c5eead51553a555705ffd4be440b76050c538ec.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/734120af2ad82465c1ba9a723c5eead51553a555705ffd4be440b76050c538ec.jpg)

![92fd0ae68020ddfdfb25cc4e05b33a0c0b5372a7726e55aa18d0e7e4355d6c58.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/92fd0ae68020ddfdfb25cc4e05b33a0c0b5372a7726e55aa18d0e7e4355d6c58.jpg)

![b28f918ee50c288cf7807b6b8a8aa6eb2cb1810c7128deef86f81a9093a6f342.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/b28f918ee50c288cf7807b6b8a8aa6eb2cb1810c7128deef86f81a9093a6f342.jpg)

![b75521e83e3fd3bfd6a6089755d4cd3d1cf12531900e524c735898f9b2e0b22f.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/b75521e83e3fd3bfd6a6089755d4cd3d1cf12531900e524c735898f9b2e0b22f.jpg)

![c6141cd48f5f19c0991378798ce5556193952bec28b4544bf1b96d60c491e7d0.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/c6141cd48f5f19c0991378798ce5556193952bec28b4544bf1b96d60c491e7d0.jpg)

![c723285fc44efdb3056facaf0a207d74dab507338edbc762ac809a1ac24dd7df.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/c723285fc44efdb3056facaf0a207d74dab507338edbc762ac809a1ac24dd7df.jpg)

![cb2c1b07415348fd6e6ce57d8e75bccf723bcfc58fe8ad3526d78ebc1c72dc47.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/images/cb2c1b07415348fd6e6ce57d8e75bccf723bcfc58fe8ad3526d78ebc1c72dc47.jpg)

### Tables

![3849a766bfb009da9e8efaf8d866a61249e72e126d2c5e1f38dbccfe044979d7.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/tables/3849a766bfb009da9e8efaf8d866a61249e72e126d2c5e1f38dbccfe044979d7.jpg)

![468cd974bb794498c5460aa92a281cfc20d28498d4d593c3203bf017cab1b8de.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/tables/468cd974bb794498c5460aa92a281cfc20d28498d4d593c3203bf017cab1b8de.jpg)

![4c25c6c986aa494e503cd5dfc5ce496efb24cbe5b14c469940e42456b55d9680.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/tables/4c25c6c986aa494e503cd5dfc5ce496efb24cbe5b14c469940e42456b55d9680.jpg)

![b9c44291a651128067533cd3b29b230f35bb2f034548555d4fd262825ff42627.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/tables/b9c44291a651128067533cd3b29b230f35bb2f034548555d4fd262825ff42627.jpg)

![cdcf4730ffa6a76d404e638cc9fbc59435a24bd00295958b561b020b046bd8fa.jpg](../icml_results/1062_Noise%20as%20a%20Natural%20Regularizer%20in%20Markov%20Decision%20Processes_%20Connecting%20Environmental%20Stochasticity%20/tables/cdcf4730ffa6a76d404e638cc9fbc59435a24bd00295958b561b020b046bd8fa.jpg)

<a id="paper-1045"></a>

## Beyond Blind Noising: Disentangled Visual Rectification for Hallucination Mitigation in MLLMs


### Images

![0bd4e6e9b397820a773211dd71190a999d584140c1cf88bd3c3d2760f61adf51.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/0bd4e6e9b397820a773211dd71190a999d584140c1cf88bd3c3d2760f61adf51.jpg)

![1d409d8a3b3c8e95c44f69aac3667a3c026421b1c97b2f14aa72f09abbe022b2.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/1d409d8a3b3c8e95c44f69aac3667a3c026421b1c97b2f14aa72f09abbe022b2.jpg)

![219016fbee29dedc0edb869e820a5c4ae50dda229ab000ff44ab47f6234113d0.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/219016fbee29dedc0edb869e820a5c4ae50dda229ab000ff44ab47f6234113d0.jpg)

![24c3cf2b9bf126a22115196038885b016227188adab4a86e8b0b21237b9e103c.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/24c3cf2b9bf126a22115196038885b016227188adab4a86e8b0b21237b9e103c.jpg)

![323fc528ab34493b18f88980b872bf1c056b6c6c4a0f11ddff220afcf52ecf36.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/323fc528ab34493b18f88980b872bf1c056b6c6c4a0f11ddff220afcf52ecf36.jpg)

![4ed7cb31a03a99593a72579d9997a65a203b5e47fdbec0b9be8b75b9f63a1ac3.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/4ed7cb31a03a99593a72579d9997a65a203b5e47fdbec0b9be8b75b9f63a1ac3.jpg)

![5264529ed533b81536a9f593ae51721b38423c9a54fe70905682e3bb44b5c0b8.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/5264529ed533b81536a9f593ae51721b38423c9a54fe70905682e3bb44b5c0b8.jpg)

![61d8145a3932f16b4a9879a89369d1a3188e1d5d09abe2ff1c23730bd4d8d7f9.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/61d8145a3932f16b4a9879a89369d1a3188e1d5d09abe2ff1c23730bd4d8d7f9.jpg)

![79fe9f8fc9f54e545c31d76e271e4b4a9341683ffd5a2228a92006041cb8f547.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/79fe9f8fc9f54e545c31d76e271e4b4a9341683ffd5a2228a92006041cb8f547.jpg)

![989d35601f718e579ab716637c96cf5c7cb382b36a6a2a65384c8ca1baa3e492.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/989d35601f718e579ab716637c96cf5c7cb382b36a6a2a65384c8ca1baa3e492.jpg)

![b839717e5aadc0a2944778448d3ba1287abc5eab54f8c821565c8893d0f058a8.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/b839717e5aadc0a2944778448d3ba1287abc5eab54f8c821565c8893d0f058a8.jpg)

![bd7c3cdb701e7d19c22678ecf716023068af182c74589ab4b6929bcdca063ffb.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/bd7c3cdb701e7d19c22678ecf716023068af182c74589ab4b6929bcdca063ffb.jpg)

![c1b8712546f6059287020a04f5ce3ee669c121502c22f03b76fc780623fc822f.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/c1b8712546f6059287020a04f5ce3ee669c121502c22f03b76fc780623fc822f.jpg)

![e0d054307db2319e4ff757793a128e8d277a53a7ff5a72a122236cb3d8931341.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/images/e0d054307db2319e4ff757793a128e8d277a53a7ff5a72a122236cb3d8931341.jpg)

### Tables

![0a820ec1506733d0351e7d29631e895e9243df94e25ef0eee5f36624e8c20d3a.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/0a820ec1506733d0351e7d29631e895e9243df94e25ef0eee5f36624e8c20d3a.jpg)

![1c5f164df8357ce7098722b7002765298da73fbd9fbf14cffd0ebb843150e6e4.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/1c5f164df8357ce7098722b7002765298da73fbd9fbf14cffd0ebb843150e6e4.jpg)

![1ce599dc0210f8ae947321237ffe0678cf479d9d180b0e523c25985322d0711b.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/1ce599dc0210f8ae947321237ffe0678cf479d9d180b0e523c25985322d0711b.jpg)

![1d21c9990b9f6dbaadf5e450144728094f621d1dac44cbba53b096125296f49a.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/1d21c9990b9f6dbaadf5e450144728094f621d1dac44cbba53b096125296f49a.jpg)

![20ca8e25ebd8291e93e7ea50e61fbd35a3e8cbe31dcc9c1aa45c785cfb452ce9.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/20ca8e25ebd8291e93e7ea50e61fbd35a3e8cbe31dcc9c1aa45c785cfb452ce9.jpg)

![2cdcb94c3b47871c43b42300e718bb3a05b3fa1bc6d3a4f50798b78f9f9e999f.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/2cdcb94c3b47871c43b42300e718bb3a05b3fa1bc6d3a4f50798b78f9f9e999f.jpg)

![3244d7fc0b852280daa06882f565b5ce4fd9781ebff99beb1113a15760437c8a.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/3244d7fc0b852280daa06882f565b5ce4fd9781ebff99beb1113a15760437c8a.jpg)

![4f7dfaf279f399262b4a66e28ddf7fd672a65aaaee12adbe791688045aa7b9e9.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/4f7dfaf279f399262b4a66e28ddf7fd672a65aaaee12adbe791688045aa7b9e9.jpg)

![50c7e3d23d3b1f820e0a22bb093b31b8746ddaeb7ec64eff6f777de607ddf347.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/50c7e3d23d3b1f820e0a22bb093b31b8746ddaeb7ec64eff6f777de607ddf347.jpg)

![62d94a6a3975b9658754ca5c2fa4c2c6a475898d2ed7b634592ceb8a74614483.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/62d94a6a3975b9658754ca5c2fa4c2c6a475898d2ed7b634592ceb8a74614483.jpg)

![695a356f05db83ae434630ab7403ede229edfa3c2308cc8c45f90ff1d6c5e327.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/695a356f05db83ae434630ab7403ede229edfa3c2308cc8c45f90ff1d6c5e327.jpg)

![71c4d2dd2b766be81f6a1d10de45ee4f8fbc9795646e3687e4b99ee5ede3505b.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/71c4d2dd2b766be81f6a1d10de45ee4f8fbc9795646e3687e4b99ee5ede3505b.jpg)

![7d5b8094796c067e8c3d8d48407a0828eaeab72f35f1775e25f983ce61509138.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/7d5b8094796c067e8c3d8d48407a0828eaeab72f35f1775e25f983ce61509138.jpg)

![8f9ffc287c3ef9d069276217d7a28b6162d3548083474bc847499aa90ff87031.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/8f9ffc287c3ef9d069276217d7a28b6162d3548083474bc847499aa90ff87031.jpg)

![979e5961bb2802bdaed6507b08015488b247a25ffc8473c4d7bca2977f49bb30.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/979e5961bb2802bdaed6507b08015488b247a25ffc8473c4d7bca2977f49bb30.jpg)

![9bd12cba3f4ba139f6e5f65d53c13bfe45e304e58420fd047b7bd18782033f5f.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/9bd12cba3f4ba139f6e5f65d53c13bfe45e304e58420fd047b7bd18782033f5f.jpg)

![9c76ad335295f781963303ad4cf5159670faab7e6657764cc279a6ae4c963498.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/9c76ad335295f781963303ad4cf5159670faab7e6657764cc279a6ae4c963498.jpg)

![9f3d1816da18c1c011eed2a099760552ae30dd5bb36fe37c39f588062c0c90e4.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/9f3d1816da18c1c011eed2a099760552ae30dd5bb36fe37c39f588062c0c90e4.jpg)

![ca97b20ed04dedd418951e20c81a4e672ef1e88b089ba76ef2262f84edf9183d.jpg](../icml_results/1063_Beyond%20Blind%20Noising_%20Disentangled%20Visual%20Rectification%20for%20Hallucination%20Mitigation%20in%20MLLMs/tables/ca97b20ed04dedd418951e20c81a4e672ef1e88b089ba76ef2262f84edf9183d.jpg)

<a id="paper-1046"></a>

## RELO: Reinforcement Learning to Localize for Visual Object Tracking


### Images

![13c84c901a267cfe2f44945e8e397296fabdbe89e65eb955ebeb5331dde4fea9.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/13c84c901a267cfe2f44945e8e397296fabdbe89e65eb955ebeb5331dde4fea9.jpg)

![297de082d7aec6715b8cd8e364023b628078cf4b6ec28b16c09209bdb782ce08.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/297de082d7aec6715b8cd8e364023b628078cf4b6ec28b16c09209bdb782ce08.jpg)

![51dfb8a63d69a5d3935e51dc05e6292b51a3b1bda7f56721a88ef6b1e6144a87.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/51dfb8a63d69a5d3935e51dc05e6292b51a3b1bda7f56721a88ef6b1e6144a87.jpg)

![8c7f201a9b96dc4ae628c020196c34aedfeb05ce4e65301d53e0f9b0a7f518ff.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/8c7f201a9b96dc4ae628c020196c34aedfeb05ce4e65301d53e0f9b0a7f518ff.jpg)

![90f0dba3344669fb5da154fe80968ee516994b2688e216290866eccef8650833.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/90f0dba3344669fb5da154fe80968ee516994b2688e216290866eccef8650833.jpg)

![c03f5fc563a9216e5567131f20f56fb9c8a48b167bfccb714074c44a2d9c0790.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/c03f5fc563a9216e5567131f20f56fb9c8a48b167bfccb714074c44a2d9c0790.jpg)

![ffc61d159c352f984f35284305f1a96e4eb9dbb2e6acbbc0726dced41eb69937.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/images/ffc61d159c352f984f35284305f1a96e4eb9dbb2e6acbbc0726dced41eb69937.jpg)

### Tables

![112c80d43a4a35941fc5249682eb0f78f0d2a07986212da2f54d1ba421782b98.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/112c80d43a4a35941fc5249682eb0f78f0d2a07986212da2f54d1ba421782b98.jpg)

![6e4d2e3b7fbcb6ab52e82ac4d8fe049ce04c6ae47d1a7a657e495777e6e17332.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/6e4d2e3b7fbcb6ab52e82ac4d8fe049ce04c6ae47d1a7a657e495777e6e17332.jpg)

![9c5dcc33ad3cb4c86e171882e85145249e8153c9247d2513c9c72f375d998a12.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/9c5dcc33ad3cb4c86e171882e85145249e8153c9247d2513c9c72f375d998a12.jpg)

![b169d2013475e99411a1a5ac055253510546f97ec7a20200de9ab4fd6bab4b57.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/b169d2013475e99411a1a5ac055253510546f97ec7a20200de9ab4fd6bab4b57.jpg)

![d2d069506afd9e980578a40acb99a222d2ebfc25c0fec9531b9b6b529085ce23.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/d2d069506afd9e980578a40acb99a222d2ebfc25c0fec9531b9b6b529085ce23.jpg)

![de17d7ebfe6bbc957fc92570c76ed5d9ea49dcec650c25284d27e0f8c9f32af8.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/de17d7ebfe6bbc957fc92570c76ed5d9ea49dcec650c25284d27e0f8c9f32af8.jpg)

![fa4ac480d983b4805db5143d9bfb8bb1a96f6d620bcff63b9be5b8a7278514e8.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/fa4ac480d983b4805db5143d9bfb8bb1a96f6d620bcff63b9be5b8a7278514e8.jpg)

![fe9867b919fb4c464846521d23a415163a434e84fe9fd390bc9e5914e584e063.jpg](../icml_results/1064_RELO_%20Reinforcement%20Learning%20to%20Localize%20for%20Visual%20Object%20Tracking/tables/fe9867b919fb4c464846521d23a415163a434e84fe9fd390bc9e5914e584e063.jpg)

<a id="paper-1047"></a>

## Mixtures of Geodesic Factor Analyzers on Riemannian Homogeneous Spaces


### Images

![02377c33ec9390f14f5a02cf08c61b55db16f4dda89e140db0833eff71a47f9a.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/02377c33ec9390f14f5a02cf08c61b55db16f4dda89e140db0833eff71a47f9a.jpg)

![0441597e118bbf92d9a6f6a2de88ea467efff9afa6dc4a27d2427c14337d2d80.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/0441597e118bbf92d9a6f6a2de88ea467efff9afa6dc4a27d2427c14337d2d80.jpg)

![295b71a695de6def87eb7150f13f9cf5294f07691e2735ecbbbe02cf742e274b.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/295b71a695de6def87eb7150f13f9cf5294f07691e2735ecbbbe02cf742e274b.jpg)

![3c41c3f5075aa4aa93abc95dd36f57b6c72757af04cba6948d61b6f534f76373.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/3c41c3f5075aa4aa93abc95dd36f57b6c72757af04cba6948d61b6f534f76373.jpg)

![6fa107e10ad7a2a826c44f7f4dc45a847e38fcd15665e7d4eb00005bc27be396.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/6fa107e10ad7a2a826c44f7f4dc45a847e38fcd15665e7d4eb00005bc27be396.jpg)

![b6dfbfa3e6f7afb8f7a3d1111b04ab1bc931e8ce2646894422e03ce970e30140.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/b6dfbfa3e6f7afb8f7a3d1111b04ab1bc931e8ce2646894422e03ce970e30140.jpg)

![cc0fcb17252801b4131d0acab5686db969ad3bdf74e296f6f11abf6b1edb60dd.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/cc0fcb17252801b4131d0acab5686db969ad3bdf74e296f6f11abf6b1edb60dd.jpg)

![d6488177f3e7b9f9b51e1fc39e1a5f928923e537e516de1f416f29fb45378851.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/d6488177f3e7b9f9b51e1fc39e1a5f928923e537e516de1f416f29fb45378851.jpg)

![e5c9a69cba0fd51618d6371edb601a60092455bcc5aab3aa644d7e9c24ae4577.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/e5c9a69cba0fd51618d6371edb601a60092455bcc5aab3aa644d7e9c24ae4577.jpg)

![f0674226a6848abb2042801df163da1cf765e653d689d22faff4a910e233e14c.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/f0674226a6848abb2042801df163da1cf765e653d689d22faff4a910e233e14c.jpg)

![f6ed80e09bf248610a999c4c44feb606dc97f674422fdfe691ccd2eedac759f3.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/images/f6ed80e09bf248610a999c4c44feb606dc97f674422fdfe691ccd2eedac759f3.jpg)

### Tables

![076b6b5356aa4029ba31dad8c90f37074c94c2112f8b8d64b9ac2dccaa8e3ed1.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/076b6b5356aa4029ba31dad8c90f37074c94c2112f8b8d64b9ac2dccaa8e3ed1.jpg)

![0d785d71b5494701a08476a43f58249385ec4d6fc454b724a6a3c2ab4837a3cd.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/0d785d71b5494701a08476a43f58249385ec4d6fc454b724a6a3c2ab4837a3cd.jpg)

![1e02fb10d823d897e14f29eba17cc2d5ca97129d2c3a450cdf295ef11fc15907.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/1e02fb10d823d897e14f29eba17cc2d5ca97129d2c3a450cdf295ef11fc15907.jpg)

![3f165a5ad44b9837764977c706adea051fd9ba6949aba3fc499ec1e72fd029b0.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/3f165a5ad44b9837764977c706adea051fd9ba6949aba3fc499ec1e72fd029b0.jpg)

![4f6159ade97b18af0631ae99bea2422810425a35e13d2ae513ae9d15acf71ca1.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/4f6159ade97b18af0631ae99bea2422810425a35e13d2ae513ae9d15acf71ca1.jpg)

![61ec2ed1db2666dbc11dc69ed38aa73be8e14531d5852981c881daeccaa3695a.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/61ec2ed1db2666dbc11dc69ed38aa73be8e14531d5852981c881daeccaa3695a.jpg)

![88293d805106107eb6fafb9deb332f362dbe538937397f3e4d53320cf83effe9.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/88293d805106107eb6fafb9deb332f362dbe538937397f3e4d53320cf83effe9.jpg)

![c2bc31dabd9daea90c06743878c27e8d0fb65541e5f70b245684cc3cc37e0c93.jpg](../icml_results/1065_Mixtures%20of%20Geodesic%20Factor%20Analyzers%20on%20Riemannian%20Homogeneous%20Spaces/tables/c2bc31dabd9daea90c06743878c27e8d0fb65541e5f70b245684cc3cc37e0c93.jpg)

<a id="paper-1048"></a>

## Beyond Token-level Supervision: Unlocking the Potential of Decoding-based Regression via Reinforcement Learning


### Images

![02ed3534acfd466764eb4fce5791e2ee2eceb6ae2c56d405f1b3512d3a94b9f8.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/02ed3534acfd466764eb4fce5791e2ee2eceb6ae2c56d405f1b3512d3a94b9f8.jpg)

![20544dda12f4a702676230210b06c4dd1756fb2c3fa753e026b4c6d2204964cf.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/20544dda12f4a702676230210b06c4dd1756fb2c3fa753e026b4c6d2204964cf.jpg)

![554f0fadb4a0e5d6efe206f3a81c1ceef7b942c706ed5c54fcf7aaae019274c9.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/554f0fadb4a0e5d6efe206f3a81c1ceef7b942c706ed5c54fcf7aaae019274c9.jpg)

![5a37f5c065b46897c9e0db06c4120d870928e79c95016dc954b017601cac1d6f.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/5a37f5c065b46897c9e0db06c4120d870928e79c95016dc954b017601cac1d6f.jpg)

![5cb2321567bfef2451cc2131ca6f4c1e52654f1f040ebaeb4f0fcfcd7d74e13f.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/5cb2321567bfef2451cc2131ca6f4c1e52654f1f040ebaeb4f0fcfcd7d74e13f.jpg)

![6191bb087d3a9c26e6cf06552cf9b226cbef1b368b166bc1dcb903046ae569dc.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/6191bb087d3a9c26e6cf06552cf9b226cbef1b368b166bc1dcb903046ae569dc.jpg)

![64e7ad7ffbd707b56693b114931b816a93c4b291567daaca2dec533113996501.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/64e7ad7ffbd707b56693b114931b816a93c4b291567daaca2dec533113996501.jpg)

![685a7661d6a2ba1ff838dd2e29e16419fbc58fb6b31d77311d779d2b4c9dba2d.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/685a7661d6a2ba1ff838dd2e29e16419fbc58fb6b31d77311d779d2b4c9dba2d.jpg)

![88998d148d3c8f43b9bbcf30318734aa725ed2d8e4c2ca7b7a76c0e415bb4404.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/88998d148d3c8f43b9bbcf30318734aa725ed2d8e4c2ca7b7a76c0e415bb4404.jpg)

![8db38d35ed99662b6bcdc84b19ecc8b5039e910f78cc0f3994928b62e10e8eea.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/8db38d35ed99662b6bcdc84b19ecc8b5039e910f78cc0f3994928b62e10e8eea.jpg)

![e7b8c1c7c9a91f50a200264fccc66444e8506801bab8e74957612510d3d895e6.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/images/e7b8c1c7c9a91f50a200264fccc66444e8506801bab8e74957612510d3d895e6.jpg)

### Tables

![0a4f4bfee58d1a201ea653b881a598394789fc53f1fe9ac5a4812c33ebe8e6a5.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/0a4f4bfee58d1a201ea653b881a598394789fc53f1fe9ac5a4812c33ebe8e6a5.jpg)

![1c9fe725b68aa30546c8e526d60f8ef1b4a9556a81d4fcee0a2b7c4a48f3811a.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/1c9fe725b68aa30546c8e526d60f8ef1b4a9556a81d4fcee0a2b7c4a48f3811a.jpg)

![2bb69e602a55a0a535d919f5a06307cc96f7b661e49229f6d34971f317159c53.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/2bb69e602a55a0a535d919f5a06307cc96f7b661e49229f6d34971f317159c53.jpg)

![305a92e797ad67d1c495a79b83fb78a28fd7fbd46f9be9f9fe99ae85ec32e892.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/305a92e797ad67d1c495a79b83fb78a28fd7fbd46f9be9f9fe99ae85ec32e892.jpg)

![353a9f94517a3b15f08f87d36282f2eaa810bc1abc9683e232fce6f7d0baf3c9.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/353a9f94517a3b15f08f87d36282f2eaa810bc1abc9683e232fce6f7d0baf3c9.jpg)

![512f7ee873ce332bb69d90c9a62cbf54fde23e36634a3a021fab04bca7387d52.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/512f7ee873ce332bb69d90c9a62cbf54fde23e36634a3a021fab04bca7387d52.jpg)

![51a4dd75563f1a7eb3f7fc89a7e9a015274f257b1abbf68946b0b85742c5ca22.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/51a4dd75563f1a7eb3f7fc89a7e9a015274f257b1abbf68946b0b85742c5ca22.jpg)

![6874dea97cfec26d0cbbce6f78168e105bfead985a0b9d683743be913ac3dec0.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/6874dea97cfec26d0cbbce6f78168e105bfead985a0b9d683743be913ac3dec0.jpg)

![770e2bcbf16f7385fdaa7ce3f4fbfc7449795339ee7b09efcbea8fb7054b0f00.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/770e2bcbf16f7385fdaa7ce3f4fbfc7449795339ee7b09efcbea8fb7054b0f00.jpg)

![a7bd00ec9b0de6b6e06895623223698504d3daa39ac037699a4964d775654a96.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/a7bd00ec9b0de6b6e06895623223698504d3daa39ac037699a4964d775654a96.jpg)

![b39bdafd734b166564c0071b827fc6fa1e9b1545e200c62e1069374e1dddb57c.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/b39bdafd734b166564c0071b827fc6fa1e9b1545e200c62e1069374e1dddb57c.jpg)

![c140a8d3292865eb64ad5ecd86982bf0336467d1989e7fe7d84434f6be33f410.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/c140a8d3292865eb64ad5ecd86982bf0336467d1989e7fe7d84434f6be33f410.jpg)

![c3fd775a7a6df93b9215ec65e09a8dff58a488e7939ef728bc84e1f9c0fc6b9a.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/c3fd775a7a6df93b9215ec65e09a8dff58a488e7939ef728bc84e1f9c0fc6b9a.jpg)

![dae29ed9375bf6560e2b1156bbda5a65eabf3563caa5e32cb0edc8bdb1152072.jpg](../icml_results/1066_Beyond%20Token-level%20Supervision_%20Unlocking%20the%20Potential%20of%20Decoding-based%20Regression%20via%20Reinforceme/tables/dae29ed9375bf6560e2b1156bbda5a65eabf3563caa5e32cb0edc8bdb1152072.jpg)

<a id="paper-1049"></a>

## OmniVideo-R1: Reinforcing Audio-visual Reasoning with Query Intention and Modality Attention


### Images

![015b900ab56e1bbb6d07265220c7f9dbf86bb710656c70c05f02723141a4b85c.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/015b900ab56e1bbb6d07265220c7f9dbf86bb710656c70c05f02723141a4b85c.jpg)

![1343fc72a4d0b5e80725d01585b573d06742a70fd7554c8bc5703dae4d4a0e1f.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/1343fc72a4d0b5e80725d01585b573d06742a70fd7554c8bc5703dae4d4a0e1f.jpg)

![22f12238dc4996c4529f3edc318ccedd43448f4324cb3f92fb58ceabfb7cbdc2.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/22f12238dc4996c4529f3edc318ccedd43448f4324cb3f92fb58ceabfb7cbdc2.jpg)

![793107bff1655ccddc28e7d5923a26b521db12e1e1d89a207b143f26ddffda5c.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/793107bff1655ccddc28e7d5923a26b521db12e1e1d89a207b143f26ddffda5c.jpg)

![e07c11fb9da3c08110752325a3544a131091a5a4795742e594531b1be12a45ce.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/e07c11fb9da3c08110752325a3544a131091a5a4795742e594531b1be12a45ce.jpg)

![e32b60c4e7838d205089ec74503a80e738dac6069497dddbeefe594475c7d7e2.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/e32b60c4e7838d205089ec74503a80e738dac6069497dddbeefe594475c7d7e2.jpg)

![fd9c242acca8cc8671f82c97c1fd182bdb54d3b9be690e067ae92006fe863ae0.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/fd9c242acca8cc8671f82c97c1fd182bdb54d3b9be690e067ae92006fe863ae0.jpg)

![fe4147dad1933ac4d31d3e4a5832f806d0193b28f4b53404f59dd30d5033c0f1.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/images/fe4147dad1933ac4d31d3e4a5832f806d0193b28f4b53404f59dd30d5033c0f1.jpg)

### Tables

![07e86d5c32b58a9f1f9ce677e58d75f42ee72b106bbd7d701c4f8daefa9bdf79.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/07e86d5c32b58a9f1f9ce677e58d75f42ee72b106bbd7d701c4f8daefa9bdf79.jpg)

![2c02b50a89efa3c60f6b4ca1cc2bd94f50185642e423aee4cb1268f2fc35114a.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/2c02b50a89efa3c60f6b4ca1cc2bd94f50185642e423aee4cb1268f2fc35114a.jpg)

![50726275bc66de74de1553bed7644224c9921db528ede7c0e35cac1dd29d26e2.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/50726275bc66de74de1553bed7644224c9921db528ede7c0e35cac1dd29d26e2.jpg)

![51aa1294fb1bde984eba5e80cd7aa369f15e2a0ea679bcecbf0ce29737573aea.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/51aa1294fb1bde984eba5e80cd7aa369f15e2a0ea679bcecbf0ce29737573aea.jpg)

![566d9c2aebc02d451e6bfc1a35730645a873d6650f25c1dba74bf52b702f0a8c.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/566d9c2aebc02d451e6bfc1a35730645a873d6650f25c1dba74bf52b702f0a8c.jpg)

![81739089e52aecc674150f78073dfb4ae7da834d39d5defd76276e66151eb78b.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/81739089e52aecc674150f78073dfb4ae7da834d39d5defd76276e66151eb78b.jpg)

![86f3a42cfd14cd5c21085a253dbfb843eb12ba8fb225a6f48998de721ef88961.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/86f3a42cfd14cd5c21085a253dbfb843eb12ba8fb225a6f48998de721ef88961.jpg)

![a227b610135285f373873d7f2380b3f871f7a2ffc65de5e4137e8dae623fa86f.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/a227b610135285f373873d7f2380b3f871f7a2ffc65de5e4137e8dae623fa86f.jpg)

![aa63964c0536c13123822818218cf81141ed73c28f5bdfb846668e1b871dfc6a.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/aa63964c0536c13123822818218cf81141ed73c28f5bdfb846668e1b871dfc6a.jpg)

![abc7e78a79b5bd1ecd75561f152f40bc4242576c8d1ce79b151fd2afd7bfed47.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/abc7e78a79b5bd1ecd75561f152f40bc4242576c8d1ce79b151fd2afd7bfed47.jpg)

![ace5974223a34304ebc016e826c31ab31bcff21d17681048362001d702047264.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/ace5974223a34304ebc016e826c31ab31bcff21d17681048362001d702047264.jpg)

![b01bde409822c3bb24c4d3f191fc5a335b7773220d42d3466b16804e8d90468e.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/b01bde409822c3bb24c4d3f191fc5a335b7773220d42d3466b16804e8d90468e.jpg)

![bcfc9c194ef73e1131b5e225374b95c35b55af22a599e83c59cb7ce3036248fc.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/bcfc9c194ef73e1131b5e225374b95c35b55af22a599e83c59cb7ce3036248fc.jpg)

![e5f47fdbe1f802d69314c193348021b54144e969a1f5950ce4218125148f4559.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/e5f47fdbe1f802d69314c193348021b54144e969a1f5950ce4218125148f4559.jpg)

![fe639104d41e40cae0cc583ad0a05ab5c80703a15d44d62d41e09f6a3e40d576.jpg](../icml_results/1067_OmniVideo-R1_%20Reinforcing%20Audio-visual%20Reasoning%20with%20Query%20Intention%20and%20Modality%20Attention/tables/fe639104d41e40cae0cc583ad0a05ab5c80703a15d44d62d41e09f6a3e40d576.jpg)

<a id="paper-1050"></a>

## DynaTok: Token-Based 4D Reconstruction from Partial Point Clouds


### Images

![18bac399862fa1fb56a7cb7927e1b52bbfa56fbdc9647cc0a725e1d69b097a8d.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/18bac399862fa1fb56a7cb7927e1b52bbfa56fbdc9647cc0a725e1d69b097a8d.jpg)

![40e52714db225f1cfe36fccf8d5845d1b49621fd36e7e89f3fd509df34347dd5.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/40e52714db225f1cfe36fccf8d5845d1b49621fd36e7e89f3fd509df34347dd5.jpg)

![4962837970622b4da1c6a81aa8abfb018bcf3e368910a223bed71147d9632f3d.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/4962837970622b4da1c6a81aa8abfb018bcf3e368910a223bed71147d9632f3d.jpg)

![6934e8307c272d73dcdda8e23550a47b5339feb116ea0936004df88af8cbc540.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/6934e8307c272d73dcdda8e23550a47b5339feb116ea0936004df88af8cbc540.jpg)

![7fb7e7bded0a1c6feea618157755be9c2e786da9508f6677782763e12c23900b.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/7fb7e7bded0a1c6feea618157755be9c2e786da9508f6677782763e12c23900b.jpg)

![9443583e284238a2695e65c27e1fb17d166cf37f0bd41855d24cb47d875cea06.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/9443583e284238a2695e65c27e1fb17d166cf37f0bd41855d24cb47d875cea06.jpg)

![eba8a27877a7ec23c313c1afff37aa4212b77ba5cdb22defa11544fa2926f668.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/eba8a27877a7ec23c313c1afff37aa4212b77ba5cdb22defa11544fa2926f668.jpg)

![f6ff8c3d34e42c2edbfd9c5ec2228cf58fc4a95036e0df7a5d221d71998e0eb9.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/images/f6ff8c3d34e42c2edbfd9c5ec2228cf58fc4a95036e0df7a5d221d71998e0eb9.jpg)

### Tables

![07cffd8c4d54e5873b768f8ec9410bd77d3aacceee501a86803054596c396f2b.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/07cffd8c4d54e5873b768f8ec9410bd77d3aacceee501a86803054596c396f2b.jpg)

![1e6be51d2124b687a921f1aa56b68976636c5d303692ec103be2f5c633708a71.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/1e6be51d2124b687a921f1aa56b68976636c5d303692ec103be2f5c633708a71.jpg)

![420b12a4458c036d078ce81eac419740f5ebbeb6976a2bb080226d7dea62a18e.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/420b12a4458c036d078ce81eac419740f5ebbeb6976a2bb080226d7dea62a18e.jpg)

![539ee94a8d1955e775f4fcbd17b440e7db05379204923b21a752a38cc336033f.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/539ee94a8d1955e775f4fcbd17b440e7db05379204923b21a752a38cc336033f.jpg)

![904ed5857f45a4192652a94c80cf999000f3668c2716f3fe551c94035c6fa530.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/904ed5857f45a4192652a94c80cf999000f3668c2716f3fe551c94035c6fa530.jpg)

![926e17fbdbf2adf2326f34aa6a9f447916af12bc51d86fca214c2c845b1aa151.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/926e17fbdbf2adf2326f34aa6a9f447916af12bc51d86fca214c2c845b1aa151.jpg)

![c017ff48f8d83f89efe93d736edc8304c6553e58d2891dc8c976bf16cdceee38.jpg](../icml_results/1068_DynaTok_%20Token-Based%204D%20Reconstruction%20from%20Partial%20Point%20Clouds/tables/c017ff48f8d83f89efe93d736edc8304c6553e58d2891dc8c976bf16cdceee38.jpg)

<a id="paper-1051"></a>

## AppWorld-UL: Benchmarking Diverse Agent-User Interactions for Tool-Use


### Images

![1a1cf99221118d6b6e6369f5b8de60a16518619c98d23749bc09216ba104c56b.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/1a1cf99221118d6b6e6369f5b8de60a16518619c98d23749bc09216ba104c56b.jpg)

![513d45a9cf668cc1c48fdaf941fae07cf8673964b5f9336dc4a8e13acc3e747c.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/513d45a9cf668cc1c48fdaf941fae07cf8673964b5f9336dc4a8e13acc3e747c.jpg)

![690daad87ef9fd7f43244941f29b213a694e3c323e2ded78c3e62bfe0d012989.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/690daad87ef9fd7f43244941f29b213a694e3c323e2ded78c3e62bfe0d012989.jpg)

![6c4eeb93ec4c0a05a5865148ab0bf0fd8b136911987bd1d470b4b2b749830def.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/6c4eeb93ec4c0a05a5865148ab0bf0fd8b136911987bd1d470b4b2b749830def.jpg)

![a59459bcd651c5447db9eedb494074de82d27e5470dbc55e4a06a734fa22cdab.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/a59459bcd651c5447db9eedb494074de82d27e5470dbc55e4a06a734fa22cdab.jpg)

![c29b4851eecd7db04e0110905f708628e0c1a2da9039524470b8b67262cad311.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/c29b4851eecd7db04e0110905f708628e0c1a2da9039524470b8b67262cad311.jpg)

![ee523c675c4851203520a528efdca61712ebcb3c16a9c66798a1b54c84a2f3fd.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/images/ee523c675c4851203520a528efdca61712ebcb3c16a9c66798a1b54c84a2f3fd.jpg)

### Tables

![15c5645d11bf790118440ad635921d9b06a137780fd8c0977a22df3b7631fd7b.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/15c5645d11bf790118440ad635921d9b06a137780fd8c0977a22df3b7631fd7b.jpg)

![3e30334cb3f0f949d981c5c078f61366f9c0edb8334c8e7d8950054cb9d6ed51.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/3e30334cb3f0f949d981c5c078f61366f9c0edb8334c8e7d8950054cb9d6ed51.jpg)

![44eccb253506601586f9ab0a8df00212fb316e387e123716f5e1d2f23e5ef6f3.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/44eccb253506601586f9ab0a8df00212fb316e387e123716f5e1d2f23e5ef6f3.jpg)

![46b37ca8589ebc43474f9c34c2d23ddfd1f2f2a599ed48c7711423555351f4e8.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/46b37ca8589ebc43474f9c34c2d23ddfd1f2f2a599ed48c7711423555351f4e8.jpg)

![7a70f9e5cafe8e941cbe470592986aec84dcf02f6ec84e23809842a73a476bba.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/7a70f9e5cafe8e941cbe470592986aec84dcf02f6ec84e23809842a73a476bba.jpg)

![850608019c8024eb6b8049522c500a61693bcb5fd902bb7759bd60bb3094987c.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/850608019c8024eb6b8049522c500a61693bcb5fd902bb7759bd60bb3094987c.jpg)

![85e777af6ced6c85045d3f61705caeb86a781969850b9936d77d12c05ceccd4f.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/85e777af6ced6c85045d3f61705caeb86a781969850b9936d77d12c05ceccd4f.jpg)

![87241a7bb53d35f5953b564b88d697f4bb28fe286e8d3722ca1169e91a127ddf.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/87241a7bb53d35f5953b564b88d697f4bb28fe286e8d3722ca1169e91a127ddf.jpg)

![8b5ed9f442d94d8a65b34864f110b5eb223529de01bfdfee5226c323af9c1664.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/8b5ed9f442d94d8a65b34864f110b5eb223529de01bfdfee5226c323af9c1664.jpg)

![99dfc13a45d31d115fce81ccd1b5129fd0b406824db46b6fa3096b34de14774a.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/99dfc13a45d31d115fce81ccd1b5129fd0b406824db46b6fa3096b34de14774a.jpg)

![ce1581694eb98c5e4b5ad2de7c0111f6dae085db8346a2d2cb1e54c89e3a18c9.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/ce1581694eb98c5e4b5ad2de7c0111f6dae085db8346a2d2cb1e54c89e3a18c9.jpg)

![d4a9cd67c6508944684cb8fe97cfebd0ba1f6b7c3318e8a154c22d08dc59ec84.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/d4a9cd67c6508944684cb8fe97cfebd0ba1f6b7c3318e8a154c22d08dc59ec84.jpg)

![f3bdf28eaae09ab6c68d276ea9c92cd2c00edecd6d660edd38596e5050f19988.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/f3bdf28eaae09ab6c68d276ea9c92cd2c00edecd6d660edd38596e5050f19988.jpg)

![f7cad28b1b11cc42d35304a8e848d9c5c8d82ff21cd69074116d15a7ffa9fb1c.jpg](../icml_results/1069_AppWorld-UL_%20Benchmarking%20Diverse%20Agent-User%20Interactions%20for%20Tool-Use/tables/f7cad28b1b11cc42d35304a8e848d9c5c8d82ff21cd69074116d15a7ffa9fb1c.jpg)

<a id="paper-1052"></a>

## A Bayesian Approach to Quantify the Uncertainty of Human Ratings in a Single-Instance Multimodal Framework


### Images

![0adbca7f94ae32a3d78673f201b0965339b5e2d98987e072b927f21f4518839f.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/images/0adbca7f94ae32a3d78673f201b0965339b5e2d98987e072b927f21f4518839f.jpg)

![5dcd01004a6b7cec2a8cdf670e8bdee89d3092c2bcdb545f3679caa18a1e90be.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/images/5dcd01004a6b7cec2a8cdf670e8bdee89d3092c2bcdb545f3679caa18a1e90be.jpg)

![d8512e5c157f62c7a2393f504017900dff1515e6456789d49fe5f83668b7fd1d.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/images/d8512e5c157f62c7a2393f504017900dff1515e6456789d49fe5f83668b7fd1d.jpg)

![fac2282cc33b1fb67fa1049c69034a35b7a806e6c9bc96b571bd6e470a5f9a06.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/images/fac2282cc33b1fb67fa1049c69034a35b7a806e6c9bc96b571bd6e470a5f9a06.jpg)

### Tables

![19b40fd63d9bc97efba44cfc367a88f7cafdfda34cface65c8b7e871c44a0fc0.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/19b40fd63d9bc97efba44cfc367a88f7cafdfda34cface65c8b7e871c44a0fc0.jpg)

![54af881291f46de1836a23af49a4eb108445035623cff93c431593750b32e794.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/54af881291f46de1836a23af49a4eb108445035623cff93c431593750b32e794.jpg)

![5851562a084332e1e930e300a12445b9c2cc3ffec74ff2d60c14a2b33b59e634.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/5851562a084332e1e930e300a12445b9c2cc3ffec74ff2d60c14a2b33b59e634.jpg)

![6302793b96a659cb7be9315424b040d31927452125e9c3ac4bb15ce39d1b0855.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/6302793b96a659cb7be9315424b040d31927452125e9c3ac4bb15ce39d1b0855.jpg)

![80ee03eaf6a6795db9603bb2b9507ecea4838eb6d5d2fb36e4ea5f1f42d1f96e.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/80ee03eaf6a6795db9603bb2b9507ecea4838eb6d5d2fb36e4ea5f1f42d1f96e.jpg)

![849ee8c01f8f93924e135d7f083fe2a5deb7019df68b8f1304a3f0a84ccc1a28.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/849ee8c01f8f93924e135d7f083fe2a5deb7019df68b8f1304a3f0a84ccc1a28.jpg)

![8cc8ec9ccc1529a8b4243858785f567287e6daef94bcb1dbcdfc507c546663dc.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/8cc8ec9ccc1529a8b4243858785f567287e6daef94bcb1dbcdfc507c546663dc.jpg)

![fb281995f1905faecfbf49ebfd2d0e6fec858835e26af92181815d23a73b54bd.jpg](../icml_results/1070_A%20Bayesian%20Approach%20to%20Quantify%20the%20Uncertainty%20of%20Human%20Ratings%20in%20a%20Single-Instance%20Multimodal%20Fra/tables/fb281995f1905faecfbf49ebfd2d0e6fec858835e26af92181815d23a73b54bd.jpg)

<a id="paper-1053"></a>

## Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning


### Images

![009f172308df16ec5ee518c21e6c349138d9f316d87632066e1fe66f6a58d105.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/009f172308df16ec5ee518c21e6c349138d9f316d87632066e1fe66f6a58d105.jpg)

![21c63203277f54cdc1b8223c1bbb71210d7cfda96ede71e06e5a308c4755a0cd.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/21c63203277f54cdc1b8223c1bbb71210d7cfda96ede71e06e5a308c4755a0cd.jpg)

![27bde695dc1c7707552d7942c8183c5987fbebe2fb60fca0a10e3d511de7b3a9.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/27bde695dc1c7707552d7942c8183c5987fbebe2fb60fca0a10e3d511de7b3a9.jpg)

![3045fd96ac9676389db541a16db72d5a89af26456aca05d21ca7278a6440d024.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/3045fd96ac9676389db541a16db72d5a89af26456aca05d21ca7278a6440d024.jpg)

![3678a33e779647657bd2048a61fa3a60a4bf19aa8752729bfe2c766f7ab84164.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/3678a33e779647657bd2048a61fa3a60a4bf19aa8752729bfe2c766f7ab84164.jpg)

![41421e5b2db4a39c773707fe4210bddb989f880ce477be11844f27dd965df4f9.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/41421e5b2db4a39c773707fe4210bddb989f880ce477be11844f27dd965df4f9.jpg)

![6285efa7a1a43ec04f259e4ed5994e795a537d21e77eb5a04219b7d8d63baede.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/6285efa7a1a43ec04f259e4ed5994e795a537d21e77eb5a04219b7d8d63baede.jpg)

![64b81e8ace43689f48fb183adb7de4a538cf2e3ffe54237183e97071c34dcce6.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/64b81e8ace43689f48fb183adb7de4a538cf2e3ffe54237183e97071c34dcce6.jpg)

![65bc3397ff758eb5380b5a595cc7e04f1df59dc25b5cdef5fbc6f731b37f82e8.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/65bc3397ff758eb5380b5a595cc7e04f1df59dc25b5cdef5fbc6f731b37f82e8.jpg)

![6d4e998f0617c3d70ed4be0cadb12294e0ef2c044f23d539fd412de3db861310.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/6d4e998f0617c3d70ed4be0cadb12294e0ef2c044f23d539fd412de3db861310.jpg)

![8507963c3336a96ec0c9aeaf44376058ba60d11652c0d704ae6244f95eb5a86c.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/8507963c3336a96ec0c9aeaf44376058ba60d11652c0d704ae6244f95eb5a86c.jpg)

![8ccf134eb442a22f939dcbef41d5b59e1921850cac0b37eea4f2d20aa3c7dfc6.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/8ccf134eb442a22f939dcbef41d5b59e1921850cac0b37eea4f2d20aa3c7dfc6.jpg)

![938d0ab0bde800a13491362cde50c9b7d9558e2d8e602008d2d0800e56812a1b.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/938d0ab0bde800a13491362cde50c9b7d9558e2d8e602008d2d0800e56812a1b.jpg)

![976383eee80180d25a5a84eb5bef35383797ecee699b085e834e5f2786b68319.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/976383eee80180d25a5a84eb5bef35383797ecee699b085e834e5f2786b68319.jpg)

![a3c6d841c7621e71cf2566caaf36022c997c5ed9536c19d689f771dc61fe8ac8.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/a3c6d841c7621e71cf2566caaf36022c997c5ed9536c19d689f771dc61fe8ac8.jpg)

![a48b0886a6960f9fe966c24b534dda1b269c600e254d64b6608fcc492db3cddf.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/a48b0886a6960f9fe966c24b534dda1b269c600e254d64b6608fcc492db3cddf.jpg)

![ae3e724cb2774785f6b77221b103ab8edd4a596d909bb11840d8510b895dd08d.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/ae3e724cb2774785f6b77221b103ab8edd4a596d909bb11840d8510b895dd08d.jpg)

![b0896a0b98457c585d659919e7f2ab3d726a038909ec199d48a2db11530e1017.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/b0896a0b98457c585d659919e7f2ab3d726a038909ec199d48a2db11530e1017.jpg)

![b9518d75b87011bd3d123bafe64fe11d13a8954575502d08652eea9617b96312.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/b9518d75b87011bd3d123bafe64fe11d13a8954575502d08652eea9617b96312.jpg)

![b9c314a422875bedb86a305e063af1d69dd23ba5489e2e71f0d8bf2af4ae977b.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/b9c314a422875bedb86a305e063af1d69dd23ba5489e2e71f0d8bf2af4ae977b.jpg)

![e0c7f90b19905478f9d906b9969bb49761a888f5bb1ca8ba70e843b5f0650771.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/e0c7f90b19905478f9d906b9969bb49761a888f5bb1ca8ba70e843b5f0650771.jpg)

![eb42ecd9859f2e43e21820c00995b15b735dd4d1896cdf77984722d1d195612a.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/eb42ecd9859f2e43e21820c00995b15b735dd4d1896cdf77984722d1d195612a.jpg)

![ef5ba08fdf04abba107c7bbb4ae7ff8ede6406a821e924a94881d481c6f254da.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/images/ef5ba08fdf04abba107c7bbb4ae7ff8ede6406a821e924a94881d481c6f254da.jpg)

### Tables

![01f09511e0746e5a187ba403a755944e8d624dad8194c4ee507a810cb197bff8.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/01f09511e0746e5a187ba403a755944e8d624dad8194c4ee507a810cb197bff8.jpg)

![086638e69513900fd58c295e2fe078363021d15f513013b6482451a83f0278b2.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/086638e69513900fd58c295e2fe078363021d15f513013b6482451a83f0278b2.jpg)

![11c99abf10f1a68676ea9136aa54b7640f0732b202da1e704fed685abb237e46.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/11c99abf10f1a68676ea9136aa54b7640f0732b202da1e704fed685abb237e46.jpg)

![5c2f297742e352a35b6ec2f53163fc25b113a59f5a08300cd50f67d8d8ebb9b8.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/5c2f297742e352a35b6ec2f53163fc25b113a59f5a08300cd50f67d8d8ebb9b8.jpg)

![835262b71c661bf7ef71c694c0f027a44e87a7035d259c896b241a2944408190.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/835262b71c661bf7ef71c694c0f027a44e87a7035d259c896b241a2944408190.jpg)

![b4db1c451fea8c74b953117308eb180036d19ebf74a499319115ae98a2650c70.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/b4db1c451fea8c74b953117308eb180036d19ebf74a499319115ae98a2650c70.jpg)

![c2a07f6abbdfab0ec0be08faa0f421bc57a1c2da5afa5057f10b7aa5f9cafd4e.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/c2a07f6abbdfab0ec0be08faa0f421bc57a1c2da5afa5057f10b7aa5f9cafd4e.jpg)

![c64bd5f24ac1332a266a409c137a66793701bdd81cb5685421dd151f85e0b3c4.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/c64bd5f24ac1332a266a409c137a66793701bdd81cb5685421dd151f85e0b3c4.jpg)

![d40ac47ea5def1714d2335766db572233e3c124ebfdb16b404e6b56d57501a23.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/d40ac47ea5def1714d2335766db572233e3c124ebfdb16b404e6b56d57501a23.jpg)

![dd80d6b78a55f145a6d69887b6d19329f47141efb8801214052cf9e261446926.jpg](../icml_results/1071_Learning%20Rate%20Scaling%20across%20LoRA%20Ranks%20and%20Transfer%20to%20Full%20Finetuning/tables/dd80d6b78a55f145a6d69887b6d19329f47141efb8801214052cf9e261446926.jpg)

<a id="paper-1054"></a>

## Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training


### Images

![0b4a891b131d91d94bfc057ef576b1d8ebb8e2dc524a14059a10ac595c1a56d5.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/0b4a891b131d91d94bfc057ef576b1d8ebb8e2dc524a14059a10ac595c1a56d5.jpg)

![12bd1e4c6865f12ea827473702041bfacb450fae148e776abf3c40a91ee55057.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/12bd1e4c6865f12ea827473702041bfacb450fae148e776abf3c40a91ee55057.jpg)

![16d2a48baaf4a38ceaa2a24171405e91d9d22c2c04afb9fea79936edf6b7803c.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/16d2a48baaf4a38ceaa2a24171405e91d9d22c2c04afb9fea79936edf6b7803c.jpg)

![28eca93ea38df8b0a9ddd75c18fd2a1dbe8725eae312e99fe6a84b868ff9e3db.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/28eca93ea38df8b0a9ddd75c18fd2a1dbe8725eae312e99fe6a84b868ff9e3db.jpg)

![3249e67d7147daef133cc7d9bd9bc63995f3af63b40d66195fe00dab11f4914d.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/3249e67d7147daef133cc7d9bd9bc63995f3af63b40d66195fe00dab11f4914d.jpg)

![412617517dba1ef573134fe4205b4d03d3730ab8d86686d7788506bbacc555ad.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/412617517dba1ef573134fe4205b4d03d3730ab8d86686d7788506bbacc555ad.jpg)

![5dcf59f759d609093a3f8561fa4b38623192e120af7ba98e2503ff0b62f979fd.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/5dcf59f759d609093a3f8561fa4b38623192e120af7ba98e2503ff0b62f979fd.jpg)

![64f0c969e62104d6235211cca4b0821d0e231e06e4a8201b1d174fa4c1fa6306.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/64f0c969e62104d6235211cca4b0821d0e231e06e4a8201b1d174fa4c1fa6306.jpg)

![7c23909621fa38871a40563a7b9a8dc4b17536890f25facdec8d813916ea1fad.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/7c23909621fa38871a40563a7b9a8dc4b17536890f25facdec8d813916ea1fad.jpg)

![8c951354ca6c316124a377de07108a1a4f216b72d44e386014471e1bb52ea365.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/8c951354ca6c316124a377de07108a1a4f216b72d44e386014471e1bb52ea365.jpg)

![9c1d4fc852ca2830732281b27e850fdedd816a71ac54022c3a58f5aa86a9493e.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/9c1d4fc852ca2830732281b27e850fdedd816a71ac54022c3a58f5aa86a9493e.jpg)

![a49098d89e37c35de9f72a3a28be2eac9bf566453ab666b9b0feb258c4a78004.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/a49098d89e37c35de9f72a3a28be2eac9bf566453ab666b9b0feb258c4a78004.jpg)

![a97c086016626305c750455464895fd9df6aec4ae928531579f44631e206886c.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/a97c086016626305c750455464895fd9df6aec4ae928531579f44631e206886c.jpg)

![b57a4989a88660e2fabb624f09d3cbdc378f6c16270cce932f775e108fba2e39.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/b57a4989a88660e2fabb624f09d3cbdc378f6c16270cce932f775e108fba2e39.jpg)

![b6e20072a84b570f20e4849d091a166d53f944527cbff92171ab5ff45f8e45e8.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/b6e20072a84b570f20e4849d091a166d53f944527cbff92171ab5ff45f8e45e8.jpg)

![bb269145eea86080866f84042ee40264ae48f85fb960776ddc759df8c897194d.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/bb269145eea86080866f84042ee40264ae48f85fb960776ddc759df8c897194d.jpg)

![bb3283fa3778c4c0b461cb1851298d5d874ec6ac3ad6a3ecaa8e46900ced0840.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/bb3283fa3778c4c0b461cb1851298d5d874ec6ac3ad6a3ecaa8e46900ced0840.jpg)

![bc7e196beacb90b3c08ccce00053de9c8aaf80aaa99cefb55c3438e166996746.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/bc7e196beacb90b3c08ccce00053de9c8aaf80aaa99cefb55c3438e166996746.jpg)

![bdddad80c3957a2a6be682c1be340216804f1dd2d1b3cc8f3072c2c2783ebb48.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/bdddad80c3957a2a6be682c1be340216804f1dd2d1b3cc8f3072c2c2783ebb48.jpg)

![cdf147295a9b0187d738ed2807e3bac585620b56501c068233e25b5c586d0187.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/cdf147295a9b0187d738ed2807e3bac585620b56501c068233e25b5c586d0187.jpg)

![ef94729d3d8b3c27fdb0c9594ccfb79393b06f9c159824156053ad3f9629bbb8.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/images/ef94729d3d8b3c27fdb0c9594ccfb79393b06f9c159824156053ad3f9629bbb8.jpg)

### Tables

![723620b7a83e60da4286d440e624f728310b0b62f10652a85a19d20dcd9930fc.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/tables/723620b7a83e60da4286d440e624f728310b0b62f10652a85a19d20dcd9930fc.jpg)

![999d6d2c0529d3aaeea040208fe5d0c9b3fad3412b6f09fa35758b715a92b51e.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/tables/999d6d2c0529d3aaeea040208fe5d0c9b3fad3412b6f09fa35758b715a92b51e.jpg)

![a825a14da55536e9a7cca91cfadb1ddd35f0007fd27404cf1dda998cc563ee71.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/tables/a825a14da55536e9a7cca91cfadb1ddd35f0007fd27404cf1dda998cc563ee71.jpg)

![d084aef745230f2c3d6253160d6cb9bb22ed26b145f792cbbe4440e31b61f780.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/tables/d084aef745230f2c3d6253160d6cb9bb22ed26b145f792cbbe4440e31b61f780.jpg)

![ead48c9605a9f9a34da5715cf080c665dcf4977d3a304320c3bbefdac479795f.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/tables/ead48c9605a9f9a34da5715cf080c665dcf4977d3a304320c3bbefdac479795f.jpg)

![f243bad7adbeb0838a3d1b0d4735ad59d9fc3801366c0d5de5ce1a8d9a95771f.jpg](../icml_results/1072_Beyond%20the%20Proxy_%20Trajectory-Distilled%20Guidance%20for%20Offline%20GFlowNet%20Training/tables/f243bad7adbeb0838a3d1b0d4735ad59d9fc3801366c0d5de5ce1a8d9a95771f.jpg)

<a id="paper-1055"></a>

## Multiview Self-Representation Learning across Heterogeneous Views


### Images

![3b3433437d1fa863924e4142750b97a62f454cf3042529b003912778c992b3e7.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/3b3433437d1fa863924e4142750b97a62f454cf3042529b003912778c992b3e7.jpg)

![436b35c94234450634377ee03aff8d44905f3648eff0e502589c3ea3797e91b0.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/436b35c94234450634377ee03aff8d44905f3648eff0e502589c3ea3797e91b0.jpg)

![4c61e99f57b6f8f6c118973322584333a0585ea35c55ede5b9eb1d8432b7ff2f.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/4c61e99f57b6f8f6c118973322584333a0585ea35c55ede5b9eb1d8432b7ff2f.jpg)

![51203baed33f364819af5e25dca28935d3650f4d2ac7ec83e0c5d4d2317750c7.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/51203baed33f364819af5e25dca28935d3650f4d2ac7ec83e0c5d4d2317750c7.jpg)

![638505137d99926b37ee6a6b829d487882b52f8099b35649bf739fd349dc34da.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/638505137d99926b37ee6a6b829d487882b52f8099b35649bf739fd349dc34da.jpg)

![63ad217a160214b48bfb688722605d487f661cef5d09459b9206ab164ada3a6f.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/63ad217a160214b48bfb688722605d487f661cef5d09459b9206ab164ada3a6f.jpg)

![6ceb7c3d3385ef8597b4a6860560eaad384927e15ac812479e47e13170a4f618.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/6ceb7c3d3385ef8597b4a6860560eaad384927e15ac812479e47e13170a4f618.jpg)

![708f313937c40fea581d01b7eaca77fdf899ea6e11c51d3d53edda99b3331137.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/708f313937c40fea581d01b7eaca77fdf899ea6e11c51d3d53edda99b3331137.jpg)

![99b5da3afd646aad6c769cc702e102f634ba41ccb747f4d8aba1a80008f798e1.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/99b5da3afd646aad6c769cc702e102f634ba41ccb747f4d8aba1a80008f798e1.jpg)

![a20517b275ee17a2c7a9afcf7029edcb232d4e540f78b5e12c92d2b673f8206f.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/a20517b275ee17a2c7a9afcf7029edcb232d4e540f78b5e12c92d2b673f8206f.jpg)

![beffc91bf54e7419fc67af23034b75795c6bd5518ba53c6fdeaddb59545f2a10.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/beffc91bf54e7419fc67af23034b75795c6bd5518ba53c6fdeaddb59545f2a10.jpg)

![c12089b86a22900ff137ee30bd1480d1401a9761e0c51c0a19efe9776f838180.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/c12089b86a22900ff137ee30bd1480d1401a9761e0c51c0a19efe9776f838180.jpg)

![cf95a661dabc7130e670fae3e7dc2f295a8e157d099f58e4f46b69874b8730b6.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/images/cf95a661dabc7130e670fae3e7dc2f295a8e157d099f58e4f46b69874b8730b6.jpg)

### Tables

![449ef0f99ba6318941394496bdea151fd0e29cc7a0771a3af4cf393af2d2d50c.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/449ef0f99ba6318941394496bdea151fd0e29cc7a0771a3af4cf393af2d2d50c.jpg)

![52b636010c6b4459cca6f4147b73f2caeabfb8fdcd9322135f0ac3cec23c01f7.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/52b636010c6b4459cca6f4147b73f2caeabfb8fdcd9322135f0ac3cec23c01f7.jpg)

![53523be056708f8bf310db6546ee7bdd095c6ef331044588b584e50e215b443f.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/53523be056708f8bf310db6546ee7bdd095c6ef331044588b584e50e215b443f.jpg)

![5df510c795002557e1afe9b285ed6d17bbf1560c143044c717aab00dfa62b891.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/5df510c795002557e1afe9b285ed6d17bbf1560c143044c717aab00dfa62b891.jpg)

![6f16f434e49df0b3e2eefd8e556f0cc4fa754dc2a1aa0e21e6c94874e5f2fff0.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/6f16f434e49df0b3e2eefd8e556f0cc4fa754dc2a1aa0e21e6c94874e5f2fff0.jpg)

![af8a4fa39f4d0b465a3cfd4d6c96962d653b3a862be1512c7ed924d7043ae1ea.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/af8a4fa39f4d0b465a3cfd4d6c96962d653b3a862be1512c7ed924d7043ae1ea.jpg)

![e8eaaf0d606b882e0b3cd8ddd565f14e081afd4ff01dc1f662a9df1a2959a7e6.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/e8eaaf0d606b882e0b3cd8ddd565f14e081afd4ff01dc1f662a9df1a2959a7e6.jpg)

![edc4e12dd548ed453f0acad5789b34cbe4d7d493028926d399a0ffce65f77db7.jpg](../icml_results/1073_Multiview%20Self-Representation%20Learning%20across%20Heterogeneous%20Views/tables/edc4e12dd548ed453f0acad5789b34cbe4d7d493028926d399a0ffce65f77db7.jpg)

<a id="paper-1056"></a>

## Discrete Tilt Matching

### Images

![1360150f8b49c6f0310a982010e791c5566c61de9d6bdac7afab81ddd0b9bb77.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/1360150f8b49c6f0310a982010e791c5566c61de9d6bdac7afab81ddd0b9bb77.jpg)

![2ce2b27d679868de87c620597488044c45c77d618070f4df39ec85b977ad0558.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/2ce2b27d679868de87c620597488044c45c77d618070f4df39ec85b977ad0558.jpg)

![3cb9cd8d9f3b8c0810c32dafa630fa0b67bd0b9ed80e39d36502ee33f651fe41.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/3cb9cd8d9f3b8c0810c32dafa630fa0b67bd0b9ed80e39d36502ee33f651fe41.jpg)

![450694d581cac33fafc72c0354096335ea0a9c6e7337b1472ec47f8b35ff535a.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/450694d581cac33fafc72c0354096335ea0a9c6e7337b1472ec47f8b35ff535a.jpg)

![5ed0ed22188bc20b3718c0dec3830c0a4ddc514ab0176350f0223f232248cc3c.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/5ed0ed22188bc20b3718c0dec3830c0a4ddc514ab0176350f0223f232248cc3c.jpg)

![720f5dc12913b6cc8862e9fee0efe2d1be54ab0a37e54a92a1d2b6f07bcc4678.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/720f5dc12913b6cc8862e9fee0efe2d1be54ab0a37e54a92a1d2b6f07bcc4678.jpg)

![8957991c06b81b0a495f788a39a985c016988c208d57b360c10588297a1add93.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/8957991c06b81b0a495f788a39a985c016988c208d57b360c10588297a1add93.jpg)

![8fcbab49e3135efb2a621a1624e2c3ede92a64cbf81256e0e835dd9cdddf110b.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/8fcbab49e3135efb2a621a1624e2c3ede92a64cbf81256e0e835dd9cdddf110b.jpg)

![954a71b15430f9e48e44398f715eaac653e464c9d4b6525f523a7056546cd90f.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/954a71b15430f9e48e44398f715eaac653e464c9d4b6525f523a7056546cd90f.jpg)

![9f892716b30d7f569e900c3060f24b1e51fb8bcd38437516b20eb1cf5cce4fa9.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/9f892716b30d7f569e900c3060f24b1e51fb8bcd38437516b20eb1cf5cce4fa9.jpg)

![a947fb48f6901ed54cc38659cd7b1cacff26560ba99b800984d2cdebf2db9910.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/a947fb48f6901ed54cc38659cd7b1cacff26560ba99b800984d2cdebf2db9910.jpg)

![a951147397fc3ecf37727f37d6e3ea78d755831b07794525548908b82c724afc.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/a951147397fc3ecf37727f37d6e3ea78d755831b07794525548908b82c724afc.jpg)

![ea7e937ada241ceb0b843ee27ef9b1ec2704cc15c5f7c2c671c604699e391cfc.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/images/ea7e937ada241ceb0b843ee27ef9b1ec2704cc15c5f7c2c671c604699e391cfc.jpg)

### Tables

![1c227160d6b39917c958ae62d8a5655ab5976caa366618a593ac6447f9d6fe1c.jpg](../icml_results/1074_Discrete%20Tilt%20Matching/tables/1c227160d6b39917c958ae62d8a5655ab5976caa366618a593ac6447f9d6fe1c.jpg)
