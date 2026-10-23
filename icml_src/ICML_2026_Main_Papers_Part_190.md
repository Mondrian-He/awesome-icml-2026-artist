# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 190 of 197

## 目录 (Table of Contents)

1. [FreeRet: MLLMs as Training-Free Retrievers](#paper-6238)
2. [DRIVE: Best Data Scheduling Practices for Reinforcement Learning with Verifiable Reward in Competitive Code Generation](#paper-6239)
3. [HSMAD: Heterophily-Driven Spectral and Manifold Learning for Graph Anomaly Detection](#paper-6240)
4. [RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents](#paper-6241)
5. [Trust It or Not: Evidential Uncertainty for Feed-Forward 3D Reconstruction with Trust3R](#paper-6242)
6. [Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation](#paper-6243)
7. [Forensic Prompting with Dual-Action Policy Optimization for Vision-Language Forgery Detection and Localization](#paper-6244)
8. [How do Humans Process AI-generated Hallucination Contents: a Neuroimaging Study](#paper-6245)
9. [ACTIVE-o3 : Empowering MLLMs with Active Perception via Pure Reinforcement Learning](#paper-6246)
10. [ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response](#paper-6247)
11. [The Geometric Reasoner: Manifold-Informed Latent Foresight Search for Long-Context Reasoning](#paper-6248)
12. [Geometric Control of Out-of-Distribution Shift in Safe Offline RL](#paper-6249)
13. [Mitigating Manifold Departure: Uncertainty-Aware Subspace Rectification for Trustworthy MLLM Decoding](#paper-6250)
14. [Efficient Adaptive Testing via Gradient Path Matching Subset Selection for AI Education](#paper-6251)
15. [Rethinking Graph Transformers as Graph Signal Denoisers: The Role of Block-Diagonal Priors](#paper-6252)
16. [Understanding the Ability of LLMs to Handle Character-Level Perturbation](#paper-6253)
17. [Training Language Model Agents to Find Vulnerabilities with CTF-Dojo](#paper-6254)
18. [On the Spectral Unreachability of Brain Graph Learning](#paper-6255)
19. [Phase-Type Variational Autoencoders for Heavy-Tailed Data](#paper-6256)
20. [Affine-Equivariant Kernel Space Encoding for NeRF Editing](#paper-6257)
21. [Can local learning match self-supervised backpropagation?](#paper-6258)
22. [Near-Optimal Dynamic Matching via Coarsening with Application to Heart Transplantation](#paper-6259)
23. [Learning Normalized Energy Models for Linear Inverse Problems](#paper-6260)
24. [PAC-Bayesian Reinforcement Learning Trains Generalizable Policies](#paper-6261)
25. [One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining](#paper-6262)
26. [HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models](#paper-6263)
27. [Conflict-Aware Adaptive Alignment for LLM Hallucination Mitigation](#paper-6264)
28. [Less Token, More Signal: MoE Expert Pruning via Critical Token Selection](#paper-6265)
29. [On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents](#paper-6266)
30. [DARC: Disagreement-Aware Alignment via Risk-Constrained Decoding](#paper-6267)
31. [MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations](#paper-6268)
32. [Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models](#paper-6269)
33. [History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction](#paper-6270)

---


<a id="paper-6238"></a>

## FreeRet: MLLMs as Training-Free Retrievers

### Images

![0df9ae299f5f10f58ba0ad72fe466ce7b730147951040399535b56a9db9abb01.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/0df9ae299f5f10f58ba0ad72fe466ce7b730147951040399535b56a9db9abb01.jpg)

![0f89fa71a05fb30526597938d2d6b0e5230bd2a9e3e4a20e22c10ca6c6f80f31.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/0f89fa71a05fb30526597938d2d6b0e5230bd2a9e3e4a20e22c10ca6c6f80f31.jpg)

![1127becdbb1836e129787829c6ce0c6476a57dc50612442c2694386c20e8d51a.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/1127becdbb1836e129787829c6ce0c6476a57dc50612442c2694386c20e8d51a.jpg)

![1c24ec3e5590ca8b4c0a399aeb591c9756b6d58eac63d2549d749cb1fcfea9d5.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/1c24ec3e5590ca8b4c0a399aeb591c9756b6d58eac63d2549d749cb1fcfea9d5.jpg)

![4815aba80162e073730312cabd52b5fe21a4574ad27e557c086a3237d498786e.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/4815aba80162e073730312cabd52b5fe21a4574ad27e557c086a3237d498786e.jpg)

![74bce590b61268d4463839fd5b4020fdef43622a08619ae1f4c10b45dc18ba3c.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/74bce590b61268d4463839fd5b4020fdef43622a08619ae1f4c10b45dc18ba3c.jpg)

![7ad45aff433b53f2a9d1e046090208f35ecc29d17d7033d5b14e79935a991b77.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/7ad45aff433b53f2a9d1e046090208f35ecc29d17d7033d5b14e79935a991b77.jpg)

![82a6255387f01a54e0fdafccca8c152b8886f9cb17f8915d4dcfa19696b84860.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/82a6255387f01a54e0fdafccca8c152b8886f9cb17f8915d4dcfa19696b84860.jpg)

![abb71e841615e38120d156e4f9698deda0a8194730e8ddfacb5649f821330571.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/abb71e841615e38120d156e4f9698deda0a8194730e8ddfacb5649f821330571.jpg)

![b2191442f68bab7ce7ef087221a935639738acb5594f8d193cb1b0edd84235a9.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/b2191442f68bab7ce7ef087221a935639738acb5594f8d193cb1b0edd84235a9.jpg)

![b9bd50996689f55478d83d72e6f7810451b34527d5da9222d606b0b5f8cd4bd8.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/b9bd50996689f55478d83d72e6f7810451b34527d5da9222d606b0b5f8cd4bd8.jpg)

![d095b6e7495a69f170ca54c829142c3f8c2c9e6cf74a191323b5f69f5d88919a.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/d095b6e7495a69f170ca54c829142c3f8c2c9e6cf74a191323b5f69f5d88919a.jpg)

![df1ff0b2f8a0bf72389f70fe23c638998e4932aeb1946672c3c90fcdb2bdddff.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/df1ff0b2f8a0bf72389f70fe23c638998e4932aeb1946672c3c90fcdb2bdddff.jpg)

![e12a972d8eed2275b7cab7e9db792210085e24ab609f1240b65be71a84fbc53c.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/images/e12a972d8eed2275b7cab7e9db792210085e24ab609f1240b65be71a84fbc53c.jpg)

### Tables

![11c730c3a78f6f85bd67060fbc5dfb9637a42e5a975f26103112a5cc9ca6d384.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/11c730c3a78f6f85bd67060fbc5dfb9637a42e5a975f26103112a5cc9ca6d384.jpg)

![334d87f6359da2922686cb828643220912c2397f0b9d4154088975dfeb5a8116.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/334d87f6359da2922686cb828643220912c2397f0b9d4154088975dfeb5a8116.jpg)

![422fcb372aa73fee233680fc0fd8cc18bd067d62803eae8d51e0700fd8fdfdd4.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/422fcb372aa73fee233680fc0fd8cc18bd067d62803eae8d51e0700fd8fdfdd4.jpg)

![5071ce9bc40b29314de931acb2e26fe6b53d9de4292e22ebfb39aef4de452c02.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/5071ce9bc40b29314de931acb2e26fe6b53d9de4292e22ebfb39aef4de452c02.jpg)

![5a815dadbc593c2806030ca22dc2e2473cab0972428a486a591ba1a8fa7b1624.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/5a815dadbc593c2806030ca22dc2e2473cab0972428a486a591ba1a8fa7b1624.jpg)

![7055c3f19583a6d8ded3e12caa27bed2c5eaf4d3df07463f4b9a25086007780a.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/7055c3f19583a6d8ded3e12caa27bed2c5eaf4d3df07463f4b9a25086007780a.jpg)

![8be6018e022f4de94957e17c3adbcfd9e1530722328de0c642fb62a59a94b2ba.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/8be6018e022f4de94957e17c3adbcfd9e1530722328de0c642fb62a59a94b2ba.jpg)

![a6c2c8dca5123463633d9ce693f9fd48a5f2c385495d3caca9df1356ced5eaf6.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/a6c2c8dca5123463633d9ce693f9fd48a5f2c385495d3caca9df1356ced5eaf6.jpg)

![fd42db49655b0df2c6164e42b9b649e0432b2544c08fc44db0331ea405dd4ba2.jpg](../icml_results/6289_PACE_%20Post-Causal%20Entropy%20Modeling%20for%20Learned%20LiDAR%20Point%20Cloud%20Compression/tables/fd42db49655b0df2c6164e42b9b649e0432b2544c08fc44db0331ea405dd4ba2.jpg)

## FreeRet: MLLMs as Training-Free Retrievers


### Images

![2e5c00b8303cb9f14cc9954003eb7cc0c27dacda7fd39793c54691769b0aa2ec.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/2e5c00b8303cb9f14cc9954003eb7cc0c27dacda7fd39793c54691769b0aa2ec.jpg)

![40a9eadc02bad582d68fe838d4f2bbea2e8496155028b8936ea08c6645f04b26.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/40a9eadc02bad582d68fe838d4f2bbea2e8496155028b8936ea08c6645f04b26.jpg)

![525369f0a1e78744c985e6ea85f27c28acb7dbd993011781b433c5dc3b2a4082.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/525369f0a1e78744c985e6ea85f27c28acb7dbd993011781b433c5dc3b2a4082.jpg)

![5a4b386f6c449af7d70ac9aa908745c50fddddf9b8d8a54f474bc99cd596de71.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/5a4b386f6c449af7d70ac9aa908745c50fddddf9b8d8a54f474bc99cd596de71.jpg)

![5f5bb3314e328bb96287fb57a0ecf5c941b046bf3ddd25ae1392b3afc1f16a09.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/5f5bb3314e328bb96287fb57a0ecf5c941b046bf3ddd25ae1392b3afc1f16a09.jpg)

![87ac4fee5598998fcf92dc8f5256cc07da0e70a035981bb90ca4f604f07f10e2.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/87ac4fee5598998fcf92dc8f5256cc07da0e70a035981bb90ca4f604f07f10e2.jpg)

![8a3952790257cf725c558f85ea641edb74ccc3cf21f3f2874d119eba79b7849f.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/8a3952790257cf725c558f85ea641edb74ccc3cf21f3f2874d119eba79b7849f.jpg)

![96bdd284e43903af42b497eba7aa2e4e799095b30124121bc880442b54397704.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/96bdd284e43903af42b497eba7aa2e4e799095b30124121bc880442b54397704.jpg)

![aaee6e13a9283df6ff1f64902d7c2bcfb3e66a9f7d8d05b54a08a4b1e72b434f.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/aaee6e13a9283df6ff1f64902d7c2bcfb3e66a9f7d8d05b54a08a4b1e72b434f.jpg)

![c6c1a3e20957ff8ad33a0349814445fcc56b121de2b1aacf4d14771b237904ca.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/c6c1a3e20957ff8ad33a0349814445fcc56b121de2b1aacf4d14771b237904ca.jpg)

![f5c53df4737fdb64caa973e888406b6a96742f0e4eb39020198cf87a11d33e27.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/images/f5c53df4737fdb64caa973e888406b6a96742f0e4eb39020198cf87a11d33e27.jpg)

### Tables

![0f2428414c7a4e52677e2f7337957dde3b3741a60e04cd0fba1192b2c10b02fa.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/0f2428414c7a4e52677e2f7337957dde3b3741a60e04cd0fba1192b2c10b02fa.jpg)

![247d0990dc209df1d41faaf18fc30df59d71aadc0fa03a272ab7dfaddf3baa89.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/247d0990dc209df1d41faaf18fc30df59d71aadc0fa03a272ab7dfaddf3baa89.jpg)

![2a945fef9b34a1cf0ce1eb6a8ef9ba5fe8f7a7ec69ee291bf3c1a98e556bae9c.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/2a945fef9b34a1cf0ce1eb6a8ef9ba5fe8f7a7ec69ee291bf3c1a98e556bae9c.jpg)

![2dde983f4eb0532fec783dea97d54c3bc3bc68e1b269e8a8923ebf8b542524fb.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/2dde983f4eb0532fec783dea97d54c3bc3bc68e1b269e8a8923ebf8b542524fb.jpg)

![762c5d02e7251a64a8313873edfd98a899e0743f6857b6a6e998a43ddb4073ea.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/762c5d02e7251a64a8313873edfd98a899e0743f6857b6a6e998a43ddb4073ea.jpg)

![7d9b317cd1f13922feea72c70b7145928dd5ddf4e64e9bf193a7f2ac266a59de.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/7d9b317cd1f13922feea72c70b7145928dd5ddf4e64e9bf193a7f2ac266a59de.jpg)

![8395da55812bf5cf67e9ca872800b19c31648320e52b30bed690d0cb93c0f4e0.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/8395da55812bf5cf67e9ca872800b19c31648320e52b30bed690d0cb93c0f4e0.jpg)

![83a163cf85e778714b092de8429a90c81533a543a97059a7be71ec4910f3d8df.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/83a163cf85e778714b092de8429a90c81533a543a97059a7be71ec4910f3d8df.jpg)

![e0a9f49c94efb56d6f4eec724470ffb1acab7c527fe5106117743f1cdc67d004.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/e0a9f49c94efb56d6f4eec724470ffb1acab7c527fe5106117743f1cdc67d004.jpg)

![e12a30079fe03b2501e9f60e1b08054519452fa3ff72e9e48ec043d253ca4743.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/e12a30079fe03b2501e9f60e1b08054519452fa3ff72e9e48ec043d253ca4743.jpg)

![e4dd27fd802baae637b1670fe5b8a9200926f021d49f00b8148e709e48439d26.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/e4dd27fd802baae637b1670fe5b8a9200926f021d49f00b8148e709e48439d26.jpg)

![fafdeed87a4c79ced909c6721da8257d7860cb7f0b8a0544903d5449b1a064cf.jpg](../icml_results/6290_FreeRet_%20MLLMs%20as%20Training-Free%20Retrievers/tables/fafdeed87a4c79ced909c6721da8257d7860cb7f0b8a0544903d5449b1a064cf.jpg)

<a id="paper-6239"></a>

## DRIVE: Best Data Scheduling Practices for Reinforcement Learning with Verifiable Reward in Competitive Code Generation


### Images

![4a566df981773f0e3995841324e2333a908966a8e65d108916909c3e065d524b.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/4a566df981773f0e3995841324e2333a908966a8e65d108916909c3e065d524b.jpg)

![6156b17b1bd5c51bfbd0a4bfd79942ad823fc7f90cb93d561c9afba7cf7aba7a.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/6156b17b1bd5c51bfbd0a4bfd79942ad823fc7f90cb93d561c9afba7cf7aba7a.jpg)

![637371d1a6353c24d0b65a13dca958fea85a5a217cc0b23abac8f122fa07fd93.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/637371d1a6353c24d0b65a13dca958fea85a5a217cc0b23abac8f122fa07fd93.jpg)

![6d11b4092ecbf685f50b6a10c6b747a7ae364b716c9576e2dd68d520fc9bd0df.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/6d11b4092ecbf685f50b6a10c6b747a7ae364b716c9576e2dd68d520fc9bd0df.jpg)

![708ec4d846fa74e061159dbcf67fe4b59e99b58e4e0992cd61e97641f740cafb.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/708ec4d846fa74e061159dbcf67fe4b59e99b58e4e0992cd61e97641f740cafb.jpg)

![75016567fc5aa6712807d4a448e2b50bcf74918e3041a5254ab3ace181984b89.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/75016567fc5aa6712807d4a448e2b50bcf74918e3041a5254ab3ace181984b89.jpg)

![822126315dfee6a8bc1b9d102ec81862b82bfe809b00d38b6a2c6e293469e0b0.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/822126315dfee6a8bc1b9d102ec81862b82bfe809b00d38b6a2c6e293469e0b0.jpg)

![a24e6cad3fa325ab0caa0438315d03e1f06d190a6b338302eb1184e365e3168a.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/a24e6cad3fa325ab0caa0438315d03e1f06d190a6b338302eb1184e365e3168a.jpg)

![b816928db3f1e54f0c5644da884e240e466b407176062e68fc1622af998c86eb.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/images/b816928db3f1e54f0c5644da884e240e466b407176062e68fc1622af998c86eb.jpg)

### Tables

![1b53f793fe570bd0bb5b85c46539d57a6d02e1d4440e2709f5801f0a1a66e6f8.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/tables/1b53f793fe570bd0bb5b85c46539d57a6d02e1d4440e2709f5801f0a1a66e6f8.jpg)

![1e456f36b980a20d750f5cb68086950ed6a4e9903e3fa16efa77b5d55e3d3e23.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/tables/1e456f36b980a20d750f5cb68086950ed6a4e9903e3fa16efa77b5d55e3d3e23.jpg)

![356d1a673c031f77c6919678f53fb398be23e7d392cac02524a0ff19407ca25b.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/tables/356d1a673c031f77c6919678f53fb398be23e7d392cac02524a0ff19407ca25b.jpg)

![8f22a2e6686e5687fab4cc7ace1bf9ac44e2acf1cf395268252c2b85edd02d07.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/tables/8f22a2e6686e5687fab4cc7ace1bf9ac44e2acf1cf395268252c2b85edd02d07.jpg)

![c38b5ba3c5b1b37ed617d8567c085c12ec270a80f3e0e6b88d4a0b09092fbc14.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/tables/c38b5ba3c5b1b37ed617d8567c085c12ec270a80f3e0e6b88d4a0b09092fbc14.jpg)

![f6580a2acdf2be4bce9752ed0a1b2aacd6da260f3b59d38ea5ad8560fd6bc0f4.jpg](../icml_results/6291_DRIVE_%20Best%20Data%20Scheduling%20Practices%20for%20Reinforcement%20Learning%20with%20Verifiable%20Reward%20in%20Competiti/tables/f6580a2acdf2be4bce9752ed0a1b2aacd6da260f3b59d38ea5ad8560fd6bc0f4.jpg)

<a id="paper-6240"></a>

## HSMAD: Heterophily-Driven Spectral and Manifold Learning for Graph Anomaly Detection


### Images

![05785d72ca0404fee1810ce05bb0bc7b58c3e71416f1bb0130057d88142191b6.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/05785d72ca0404fee1810ce05bb0bc7b58c3e71416f1bb0130057d88142191b6.jpg)

![0cb1278e36ba154dc8f86a07f957335f9d9d65fa9369c195fe4965790a48d2a5.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/0cb1278e36ba154dc8f86a07f957335f9d9d65fa9369c195fe4965790a48d2a5.jpg)

![1ec1777b0095073c4e5648c2c99cd83d83791d33f70e25903a52419f4ae7beb9.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/1ec1777b0095073c4e5648c2c99cd83d83791d33f70e25903a52419f4ae7beb9.jpg)

![224fabf15aface81a635ff4b8f40b1dde4b947a9ba456f3ec594e84c1011f3fc.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/224fabf15aface81a635ff4b8f40b1dde4b947a9ba456f3ec594e84c1011f3fc.jpg)

![36b8ccdca80347005437b70b08760f2d4c25028412ed0d026942ead0082c4c60.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/36b8ccdca80347005437b70b08760f2d4c25028412ed0d026942ead0082c4c60.jpg)

![5d4c0bbe37824f560e485d3b0c51920f1563dda86aaa4a0a6a4f64c9803a8507.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/5d4c0bbe37824f560e485d3b0c51920f1563dda86aaa4a0a6a4f64c9803a8507.jpg)

![81259c3652232aa07db3b0a38c62c4cb57a027d795397a0de502c9bf2aeb8fe2.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/81259c3652232aa07db3b0a38c62c4cb57a027d795397a0de502c9bf2aeb8fe2.jpg)

![a0ad93a3f49c6f9780ba7f655831fb211e4bffbf249c45e62a8d0fe5897b0ac3.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/a0ad93a3f49c6f9780ba7f655831fb211e4bffbf249c45e62a8d0fe5897b0ac3.jpg)

![be82d7d33e276d6a7895b5c05ad359f1bf98d129179e963eb63cd22b4e59c2d8.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/be82d7d33e276d6a7895b5c05ad359f1bf98d129179e963eb63cd22b4e59c2d8.jpg)

![cf976b4edfa309d4b453b1be83b50f964c350cd88ab2ff2828879af49c4cda84.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/cf976b4edfa309d4b453b1be83b50f964c350cd88ab2ff2828879af49c4cda84.jpg)

![feede01df8e6746b78259eafbeeac039ead83c4515297d2d2c6b881727b66b97.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/images/feede01df8e6746b78259eafbeeac039ead83c4515297d2d2c6b881727b66b97.jpg)

### Tables

![00ad52784edb3f1a5b10abb6feddef19e9df6e1004c7dac9d9285f4e880b815d.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/tables/00ad52784edb3f1a5b10abb6feddef19e9df6e1004c7dac9d9285f4e880b815d.jpg)

![31932a017c1d3d1c8a6513fa23e286f44966153eb0ba2478ff108f9daf3ded1a.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/tables/31932a017c1d3d1c8a6513fa23e286f44966153eb0ba2478ff108f9daf3ded1a.jpg)

![47295d723a12a3147ac645f308a0a9822e74872b63056c71d6f6bae7a478573e.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/tables/47295d723a12a3147ac645f308a0a9822e74872b63056c71d6f6bae7a478573e.jpg)

![8cdf5a9ffc5352873fd168e51c9314b7c0dd073e806a5f549afcae7818ea499e.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/tables/8cdf5a9ffc5352873fd168e51c9314b7c0dd073e806a5f549afcae7818ea499e.jpg)

![c772ee766f666b74087addcc794ba8efbf66f0f0d3209fca46c5d35bd0d37e81.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/tables/c772ee766f666b74087addcc794ba8efbf66f0f0d3209fca46c5d35bd0d37e81.jpg)

![d90d344cc27e9263e1073dc62f4fa0585449b76e05324bb4a409a5861518d18d.jpg](../icml_results/6292_HSMAD_%20Heterophily-Driven%20Spectral%20and%20Manifold%20Learning%20for%20Graph%20Anomaly%20Detection/tables/d90d344cc27e9263e1073dc62f4fa0585449b76e05324bb4a409a5861518d18d.jpg)

<a id="paper-6241"></a>

## RE-TRAC: REcursive TRAjectory Compression for Deep Search Agents


### Images

![2e3d98fb1241563c963c840e19721b72414a0508d52fcc75f2e4fbb207ec3681.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/2e3d98fb1241563c963c840e19721b72414a0508d52fcc75f2e4fbb207ec3681.jpg)

![32da14ba341bb3a8f2743e39661cba43fe7c0ef47ef2727e07e9642b7733f538.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/32da14ba341bb3a8f2743e39661cba43fe7c0ef47ef2727e07e9642b7733f538.jpg)

![45d05c31d6d0cbe1923029ed233118098f61c2512305e1095cf30c10222d108b.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/45d05c31d6d0cbe1923029ed233118098f61c2512305e1095cf30c10222d108b.jpg)

![9789309c30b5edfc84a9bfedc760ffe3ca89741e3b8eded16134e27b2fb31ca9.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/9789309c30b5edfc84a9bfedc760ffe3ca89741e3b8eded16134e27b2fb31ca9.jpg)

![a40d46b53d26b5d7b490c4c02cb2ab409abdd4de599f5f2ece3f7221b623348e.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/a40d46b53d26b5d7b490c4c02cb2ab409abdd4de599f5f2ece3f7221b623348e.jpg)

![caed0bf183b827107394e337038b451106177a1dee8d78db4430c52b0eae2358.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/caed0bf183b827107394e337038b451106177a1dee8d78db4430c52b0eae2358.jpg)

![de058633bc64e33166fe46f1302a99fc22ff2841a1608a9d4999d4e02cb407e0.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/de058633bc64e33166fe46f1302a99fc22ff2841a1608a9d4999d4e02cb407e0.jpg)

![effa6eca1838f791a9fcbd1bf3d0feb1c271efc12841c1812129b0061fa2d98d.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/effa6eca1838f791a9fcbd1bf3d0feb1c271efc12841c1812129b0061fa2d98d.jpg)

![f91fc017e6781e180f4bd813e4ac2378554fe09a1e3000dcecb3f8c2c6f7cbbf.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/images/f91fc017e6781e180f4bd813e4ac2378554fe09a1e3000dcecb3f8c2c6f7cbbf.jpg)

### Tables

![077232a74f20469d6e02897d32015c58b25e6a61cf39292734c8ea987ce84e32.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/077232a74f20469d6e02897d32015c58b25e6a61cf39292734c8ea987ce84e32.jpg)

![1ffa83c93881c7ab36ddd15b13848fe990e476a3326be18f02badcaaff31b608.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/1ffa83c93881c7ab36ddd15b13848fe990e476a3326be18f02badcaaff31b608.jpg)

![21e30253813b86805044e3c80fefb62818510e3dee61a625c8a8a8081f330afd.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/21e30253813b86805044e3c80fefb62818510e3dee61a625c8a8a8081f330afd.jpg)

![2d7bcd2de593f0934fc00331f5bd52732742fb12afff546a349fd299323ce5a4.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/2d7bcd2de593f0934fc00331f5bd52732742fb12afff546a349fd299323ce5a4.jpg)

![2ed3eab85019082486f1b2d75eac4c433c10eada220ee3f512110ea3068ecb76.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/2ed3eab85019082486f1b2d75eac4c433c10eada220ee3f512110ea3068ecb76.jpg)

![338cbc552e9c916979c32e1963edbf04e12c530af08fb8feb3541835b2a217f5.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/338cbc552e9c916979c32e1963edbf04e12c530af08fb8feb3541835b2a217f5.jpg)

![601665d6d91f13d2c16d7ebaf3193227ce05113658ecaf93d0ba3ffa4c1daaca.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/601665d6d91f13d2c16d7ebaf3193227ce05113658ecaf93d0ba3ffa4c1daaca.jpg)

![6da75948071528854ecd2a08931e7d5c5a1a1564482b0d24123ff85f0eb78e0e.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/6da75948071528854ecd2a08931e7d5c5a1a1564482b0d24123ff85f0eb78e0e.jpg)

![7169b7ea99b3348514c2eab1e36c8c08f9a38eb97acfe1230e5ba6cc10269ad2.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/7169b7ea99b3348514c2eab1e36c8c08f9a38eb97acfe1230e5ba6cc10269ad2.jpg)

![7cbe4ab3ef8cc5f05fb5a6544e168a68ee4f80e34c2259ee9f8a0fda564fbea8.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/7cbe4ab3ef8cc5f05fb5a6544e168a68ee4f80e34c2259ee9f8a0fda564fbea8.jpg)

![88a8a81f36219f43fbaf4abf929051013c189034d243a979bfcb7e922273d214.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/88a8a81f36219f43fbaf4abf929051013c189034d243a979bfcb7e922273d214.jpg)

![8a004fe37bae88544015de27794c3c6acbf09d5ad3e217ed5e2fe70345b7ca38.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/8a004fe37bae88544015de27794c3c6acbf09d5ad3e217ed5e2fe70345b7ca38.jpg)

![a3e0bdbcd6230319ebd0b73cd9c45330347cb41703827279e0149c693e47bd45.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/a3e0bdbcd6230319ebd0b73cd9c45330347cb41703827279e0149c693e47bd45.jpg)

![a93dbd547dfaf90978c61eb11ad5d2cc144f3e273f69f7e754fcc751449479a2.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/a93dbd547dfaf90978c61eb11ad5d2cc144f3e273f69f7e754fcc751449479a2.jpg)

![a9af035fea80fb4f9cd8ac23c518922a15ec47fee38fdd3a90bf739f769ed00f.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/a9af035fea80fb4f9cd8ac23c518922a15ec47fee38fdd3a90bf739f769ed00f.jpg)

![ad7aa4f48462e46ea87ce8e5343151430475ebdff36b97c2cd1ec5295cc0554d.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/ad7aa4f48462e46ea87ce8e5343151430475ebdff36b97c2cd1ec5295cc0554d.jpg)

![bc98e675a206927e2aeb7bf0c3195b2ec73ed59c3cc450e28bebeaa5a5558d35.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/bc98e675a206927e2aeb7bf0c3195b2ec73ed59c3cc450e28bebeaa5a5558d35.jpg)

![e211604d3ecd29e7489e7340b4a59beabc09985e7017e873babc3a95a460a633.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/e211604d3ecd29e7489e7340b4a59beabc09985e7017e873babc3a95a460a633.jpg)

![ea8155640db510987aeccb4f368067ea923c8d182a9c885ad5216f2e844879dd.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/ea8155640db510987aeccb4f368067ea923c8d182a9c885ad5216f2e844879dd.jpg)

![f4a971309e0fb552271e081500f49dc76e4c4daf0dbe31fded2aa8d84d33e38e.jpg](../icml_results/6293_RE-TRAC_%20REcursive%20TRAjectory%20Compression%20for%20Deep%20Search%20Agents/tables/f4a971309e0fb552271e081500f49dc76e4c4daf0dbe31fded2aa8d84d33e38e.jpg)

<a id="paper-6242"></a>

## Trust It or Not: Evidential Uncertainty for Feed-Forward 3D Reconstruction with Trust3R


### Images

![09c497982eb11aa4c69b5d18151b2b7017eaf4df8a6a211898f4f46a76ca3b26.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/images/09c497982eb11aa4c69b5d18151b2b7017eaf4df8a6a211898f4f46a76ca3b26.jpg)

![6578dc40f5ad5e33cb0c97a2a44c8a7a16ebb3073e6036b3e84b711df0aab768.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/images/6578dc40f5ad5e33cb0c97a2a44c8a7a16ebb3073e6036b3e84b711df0aab768.jpg)

![798e22871b537eea5c92e5618959f8bdacd48c26165d97fb2e33b316ffd39577.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/images/798e22871b537eea5c92e5618959f8bdacd48c26165d97fb2e33b316ffd39577.jpg)

![9c25cee48b863ce96c74fd1007793a83cf7c40a532673a49b101e3f157222970.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/images/9c25cee48b863ce96c74fd1007793a83cf7c40a532673a49b101e3f157222970.jpg)

![9e179b53ae526f18f1a56c747d692492aba04dc2edfe24bc64497e5a0a3000a5.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/images/9e179b53ae526f18f1a56c747d692492aba04dc2edfe24bc64497e5a0a3000a5.jpg)

![f84d2352ed0e016ad03f49f09b201ae75413a4b2db206c49e1b11bac0050d314.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/images/f84d2352ed0e016ad03f49f09b201ae75413a4b2db206c49e1b11bac0050d314.jpg)

### Tables

![0d28e0b32d781bba576997fd50eb00a6bc8860fb05892debff6632ac48d303e4.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/0d28e0b32d781bba576997fd50eb00a6bc8860fb05892debff6632ac48d303e4.jpg)

![1d15da0ed508c987347ca59ec7ed90d601b8a7330914d5be22276eaa17e4d123.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/1d15da0ed508c987347ca59ec7ed90d601b8a7330914d5be22276eaa17e4d123.jpg)

![20b29a031457a5c0e2e81501cdb24a5b8a1232d582c2b44ce94b56c7c24a10b7.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/20b29a031457a5c0e2e81501cdb24a5b8a1232d582c2b44ce94b56c7c24a10b7.jpg)

![348223b5c4ff8a448f79443fd433f18538cdec5373d1bc2962d521e8fe168555.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/348223b5c4ff8a448f79443fd433f18538cdec5373d1bc2962d521e8fe168555.jpg)

![46c2c43a709441f5b3c2ba2dc313af3b17143e80006320fde7dcf76cf4bc0413.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/46c2c43a709441f5b3c2ba2dc313af3b17143e80006320fde7dcf76cf4bc0413.jpg)

![4a4be59c9e9d6da74ad9de2efd991d87ced2594e838cd5fd6fcbe5ba025c197c.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/4a4be59c9e9d6da74ad9de2efd991d87ced2594e838cd5fd6fcbe5ba025c197c.jpg)

![4eca207d047e6857adf33a1b07b6da92aee5fd52b5a380bb9694304d165b932c.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/4eca207d047e6857adf33a1b07b6da92aee5fd52b5a380bb9694304d165b932c.jpg)

![5f14dcbed37593b5a51f01728f8a27516e00436c9c730aaba1fbdc2e92c256b9.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/5f14dcbed37593b5a51f01728f8a27516e00436c9c730aaba1fbdc2e92c256b9.jpg)

![61384271758f478f5ec6609b3f91089c16e66f4057c5ac6e47490bdc5e862eee.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/61384271758f478f5ec6609b3f91089c16e66f4057c5ac6e47490bdc5e862eee.jpg)

![6a9fe41777ad66ce77e3b4d02d7a068f507d640670976e65bc371bfd64d8ee8a.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/6a9fe41777ad66ce77e3b4d02d7a068f507d640670976e65bc371bfd64d8ee8a.jpg)

![79135e635d9cc6717618b375973abc8c24f506780c1d9237f75d1bd39f107907.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/79135e635d9cc6717618b375973abc8c24f506780c1d9237f75d1bd39f107907.jpg)

![8dc67d528e64b68b67e3cea056152cbe88b2c5e3725ab59c511230c44eed1a2b.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/8dc67d528e64b68b67e3cea056152cbe88b2c5e3725ab59c511230c44eed1a2b.jpg)

![c15e7f364ee27d9f46fbb4fbfb56e524aed4c105bfeb3fec6fde1a1d342127c7.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/c15e7f364ee27d9f46fbb4fbfb56e524aed4c105bfeb3fec6fde1a1d342127c7.jpg)

![eefef5b1393f43bbf88cd11aa49624cdf0aa42990a262913ed57092382ef0bbf.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/eefef5b1393f43bbf88cd11aa49624cdf0aa42990a262913ed57092382ef0bbf.jpg)

![f471c0a63d7fae34356fa87b0cca632f375c74a58a69ba417118c7d9091fbc13.jpg](../icml_results/6294_Trust%20It%20or%20Not_%20Evidential%20Uncertainty%20for%20Feed-Forward%203D%20Reconstruction%20with%20Trust3R/tables/f471c0a63d7fae34356fa87b0cca632f375c74a58a69ba417118c7d9091fbc13.jpg)

<a id="paper-6243"></a>

## Causal Forcing: Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation


### Images

![1470d60ecab10ec6ee9497016510c5fb4c7972079e3511a8f7d57d67b1bc1680.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/1470d60ecab10ec6ee9497016510c5fb4c7972079e3511a8f7d57d67b1bc1680.jpg)

![59b95d809c90424753b6569a839f648094b90adc5c12dbcd4088e61cd06bf974.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/59b95d809c90424753b6569a839f648094b90adc5c12dbcd4088e61cd06bf974.jpg)

![853ca91a0b74fa4eaa79ea699fe6de5719beef08882fd5c1ccef84da4c639c1d.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/853ca91a0b74fa4eaa79ea699fe6de5719beef08882fd5c1ccef84da4c639c1d.jpg)

![97faad8e13af9d83f1d4e05439f5d10b4f28689438b46f7ef0bbc0282261100d.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/97faad8e13af9d83f1d4e05439f5d10b4f28689438b46f7ef0bbc0282261100d.jpg)

![c0613d0c8d43227e61543c073f2d5ad1dac328e6f5acdbe17854a9a1cc8df0a8.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/c0613d0c8d43227e61543c073f2d5ad1dac328e6f5acdbe17854a9a1cc8df0a8.jpg)

![c6781c676f78ae9e7efe6aafe218ad6a84fc0d05de73d40977884e48df1ffc90.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/c6781c676f78ae9e7efe6aafe218ad6a84fc0d05de73d40977884e48df1ffc90.jpg)

![d376513c1bee804a5a54bfbed8827ea3f49b0f9e7a6509db82865e47bf0da4ce.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/d376513c1bee804a5a54bfbed8827ea3f49b0f9e7a6509db82865e47bf0da4ce.jpg)

![e4f45a049fcad3e5a37a1b2c262ff30cc083ec5c796e9b8b432a8d4543c2496d.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/e4f45a049fcad3e5a37a1b2c262ff30cc083ec5c796e9b8b432a8d4543c2496d.jpg)

![f80ed29f4f98c9d7e326f86d2e54fee8bb56851954de20387ffa1ad4d17870be.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/f80ed29f4f98c9d7e326f86d2e54fee8bb56851954de20387ffa1ad4d17870be.jpg)

![f8a8b5e47a987c8eef630df8d8d11cb1c49aa02bf3fab851643ef987d18669f4.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/images/f8a8b5e47a987c8eef630df8d8d11cb1c49aa02bf3fab851643ef987d18669f4.jpg)

### Tables

![c25483b47227c151a392cc53c445d7a548c3e0551176c3ac82805a1a8683808a.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/tables/c25483b47227c151a392cc53c445d7a548c3e0551176c3ac82805a1a8683808a.jpg)

![d6a44b8b5544206d5a1c29200aa6ece271061d69911d5688f24a6558d8e8282a.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/tables/d6a44b8b5544206d5a1c29200aa6ece271061d69911d5688f24a6558d8e8282a.jpg)

![eaa8ef4b2f541a5ac879441377566ffc484ec0e657d34f8fc3f3ff5d8a7ee8d0.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/tables/eaa8ef4b2f541a5ac879441377566ffc484ec0e657d34f8fc3f3ff5d8a7ee8d0.jpg)

![f4baf4ac133aa624951c6a8e6790527a21f972faa0da79290064b0750ba3912b.jpg](../icml_results/6295_Causal%20Forcing_%20Autoregressive%20Diffusion%20Distillation%20Done%20Right%20for%20High-Quality%20Real-Time%20Interact/tables/f4baf4ac133aa624951c6a8e6790527a21f972faa0da79290064b0750ba3912b.jpg)

<a id="paper-6244"></a>

## Forensic Prompting with Dual-Action Policy Optimization for Vision-Language Forgery Detection and Localization


### Images

![15009826f35a18021dea2fa0edd5d0f6a88573543b61daa8b631b18c4df85c0f.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/15009826f35a18021dea2fa0edd5d0f6a88573543b61daa8b631b18c4df85c0f.jpg)

![31e1df007409f78651f9e695bf4426f101014c171a9abb6cf2a841965a758f7c.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/31e1df007409f78651f9e695bf4426f101014c171a9abb6cf2a841965a758f7c.jpg)

![4ac8d5567ded2b3489e3ee19d33be22795ed0e87887608221f1dda66774486be.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/4ac8d5567ded2b3489e3ee19d33be22795ed0e87887608221f1dda66774486be.jpg)

![54de481b66084da21683040bfe2a92b633364836eb729fc7660996e820d5aaca.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/54de481b66084da21683040bfe2a92b633364836eb729fc7660996e820d5aaca.jpg)

![6a0f5c192974cb246032ed349f9104a44709b05c152b013e526a56f997f52605.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/6a0f5c192974cb246032ed349f9104a44709b05c152b013e526a56f997f52605.jpg)

![7a812006dc7d01a92ded8cf3c62ab91b0a4d23ddda8513abab6dc8e7acff1c3c.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/7a812006dc7d01a92ded8cf3c62ab91b0a4d23ddda8513abab6dc8e7acff1c3c.jpg)

![83afc0dfe4057b1ea2db3e11ea6923d7d023324d420f58b5ac4c1d830c26af74.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/images/83afc0dfe4057b1ea2db3e11ea6923d7d023324d420f58b5ac4c1d830c26af74.jpg)

### Tables

![3c9904a0694b639b995514516171aeecba42a2a99e8f803890fa83ad7904573b.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/3c9904a0694b639b995514516171aeecba42a2a99e8f803890fa83ad7904573b.jpg)

![4961768eb78df39c0507980505c92ede023015b7f2a0d3b6c1fbd1a7faea0a23.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/4961768eb78df39c0507980505c92ede023015b7f2a0d3b6c1fbd1a7faea0a23.jpg)

![67372202c657e8203f3372e0e42881ec897e1a9f2262848d352efa7779cbe3e2.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/67372202c657e8203f3372e0e42881ec897e1a9f2262848d352efa7779cbe3e2.jpg)

![81ab8a518fb05eeac75581fbe772be0ceb3e319f611dac48711131e045926ba7.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/81ab8a518fb05eeac75581fbe772be0ceb3e319f611dac48711131e045926ba7.jpg)

![a3206eb1da5ea553e9720bad1abba2dceb6694a3400057c03150c0c12fee027d.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/a3206eb1da5ea553e9720bad1abba2dceb6694a3400057c03150c0c12fee027d.jpg)

![b14280f451f05723390f978d9498f5ff935a7a83cb3ea5a703cbe3524c0c3a6c.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/b14280f451f05723390f978d9498f5ff935a7a83cb3ea5a703cbe3524c0c3a6c.jpg)

![cccf936c95d087364ef6795a9009f91523bc9417a205aade52514e2569457504.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/cccf936c95d087364ef6795a9009f91523bc9417a205aade52514e2569457504.jpg)

![d1213244f31bdc5db2bba210697d094c2556fbace61f1112abd0f0d8b94681c3.jpg](../icml_results/6296_Forensic%20Prompting%20with%20Dual-Action%20Policy%20Optimization%20for%20Vision-Language%20Forgery%20Detection%20and%20Lo/tables/d1213244f31bdc5db2bba210697d094c2556fbace61f1112abd0f0d8b94681c3.jpg)

<a id="paper-6245"></a>

## How do Humans Process AI-generated Hallucination Contents: a Neuroimaging Study


### Images

![3f5c345e445cc4bc90eb68745fa44b8cf20f133d1c8af1e7a3dd30d4db8ab01e.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/images/3f5c345e445cc4bc90eb68745fa44b8cf20f133d1c8af1e7a3dd30d4db8ab01e.jpg)

![87f61dea4978e923350fd6c479c30a0d46d5f86f5fc09723da9b27bee53c6a56.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/images/87f61dea4978e923350fd6c479c30a0d46d5f86f5fc09723da9b27bee53c6a56.jpg)

### Tables

![140e9360733c00b4994aa7b68ba31d147f6b93df75218a3f7ecc32199def23dc.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/140e9360733c00b4994aa7b68ba31d147f6b93df75218a3f7ecc32199def23dc.jpg)

![17eca22061f7ac4495a2c9754f04b948a3f2c55850c4f387af48d9a3e54e0361.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/17eca22061f7ac4495a2c9754f04b948a3f2c55850c4f387af48d9a3e54e0361.jpg)

![1a1fb929a1cb4a5e262b5a69679a0d40a5134bdc55896d59c1b207a91660a7bf.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/1a1fb929a1cb4a5e262b5a69679a0d40a5134bdc55896d59c1b207a91660a7bf.jpg)

![51e8801c61d9d4ba0fafbfca6271fa0c96c1787e3d9553855c37eb9cd1e7f0d8.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/51e8801c61d9d4ba0fafbfca6271fa0c96c1787e3d9553855c37eb9cd1e7f0d8.jpg)

![5a3500b30d1de3f6416171d1f03f9fb5385670163cadf1f01f69ca0d1de95cc5.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/5a3500b30d1de3f6416171d1f03f9fb5385670163cadf1f01f69ca0d1de95cc5.jpg)

![7ee41a61c9dbb672b1e4bccdccd9f85dfc9d9e3c3bda93b48555a8ab33e5fcc3.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/7ee41a61c9dbb672b1e4bccdccd9f85dfc9d9e3c3bda93b48555a8ab33e5fcc3.jpg)

![925d08c5ce2b0fd2c47ebf814a664252b635d8ee077bfc5889ee26d60c7deaeb.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/925d08c5ce2b0fd2c47ebf814a664252b635d8ee077bfc5889ee26d60c7deaeb.jpg)

![c1e6a2c3f03c23c9f6da094a405c8c9e1b42d0fb31ee0263d6ad690f0a4ece62.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/c1e6a2c3f03c23c9f6da094a405c8c9e1b42d0fb31ee0263d6ad690f0a4ece62.jpg)

![f610e1fc5037e2da19a29386f8db43990cb5344dc30a155b0e8740d072505a1c.jpg](../icml_results/6297_How%20do%20Humans%20Process%20AI-generated%20Hallucination%20Contents_%20a%20Neuroimaging%20Study/tables/f610e1fc5037e2da19a29386f8db43990cb5344dc30a155b0e8740d072505a1c.jpg)

<a id="paper-6246"></a>

## ACTIVE-o3 : Empowering MLLMs with Active Perception via Pure Reinforcement Learning


### Images

![03d006459fe4700c52b622e2c3b950402a787adcbfe4c747fb0565091b5dae20.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/03d006459fe4700c52b622e2c3b950402a787adcbfe4c747fb0565091b5dae20.jpg)

![06568cde86afd198602f1862a7e53125fd8ad197b66c404d54d0aad125d4bd4a.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/06568cde86afd198602f1862a7e53125fd8ad197b66c404d54d0aad125d4bd4a.jpg)

![069eb8696c707bfa6556b74cb2994a531d0563c4fce2a876957d72e2a10d5212.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/069eb8696c707bfa6556b74cb2994a531d0563c4fce2a876957d72e2a10d5212.jpg)

![11ecde7ed1f9d7b94e01807db5f320c1685ae7ed1182a8a305910ae8270fdfdb.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/11ecde7ed1f9d7b94e01807db5f320c1685ae7ed1182a8a305910ae8270fdfdb.jpg)

![1edf484e9d9128c380f3dafb22e2c8e2239e316daef5b3c01225bb2ad6d69d28.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/1edf484e9d9128c380f3dafb22e2c8e2239e316daef5b3c01225bb2ad6d69d28.jpg)

![28959333f55e712b515822ea43775eb2cbcd8ba7b39d24acd6a7935be5d4c748.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/28959333f55e712b515822ea43775eb2cbcd8ba7b39d24acd6a7935be5d4c748.jpg)

![2d04738c4d0239de5e266e3536a2dd31379dc9a75498d59e64c9a724c9aa5ba8.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/2d04738c4d0239de5e266e3536a2dd31379dc9a75498d59e64c9a724c9aa5ba8.jpg)

![367e7b640ce542767a565b912a1cc36559bc1a3135cd7b947583edde42004a21.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/367e7b640ce542767a565b912a1cc36559bc1a3135cd7b947583edde42004a21.jpg)

![38cddc9c4586fb7281be43a4e2d3ef235864f5f9df7419a1321b380ba9bd2c94.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/38cddc9c4586fb7281be43a4e2d3ef235864f5f9df7419a1321b380ba9bd2c94.jpg)

![3ec840021a5e48a3de72646aed4fc0d1c77345da4d820f9b6e4beeda9df1c0c0.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/3ec840021a5e48a3de72646aed4fc0d1c77345da4d820f9b6e4beeda9df1c0c0.jpg)

![5a1827e4b547fe68921083eec848ab6f30c27fc7c7de2fc3740328ab7a81f934.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/5a1827e4b547fe68921083eec848ab6f30c27fc7c7de2fc3740328ab7a81f934.jpg)

![5b3991ec9b4606c577c4c18dd403dc4ea6d1267204884a40a6a97b60bfddff4c.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/5b3991ec9b4606c577c4c18dd403dc4ea6d1267204884a40a6a97b60bfddff4c.jpg)

![6965f451a792597a86decc41149eaf0840e4755bdfe781a1fdaa793e083dc4ee.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/6965f451a792597a86decc41149eaf0840e4755bdfe781a1fdaa793e083dc4ee.jpg)

![8f307f5a42e4acffeeb2723c5fec3e7dc2cb21c53e1e338949850b070bed248c.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/8f307f5a42e4acffeeb2723c5fec3e7dc2cb21c53e1e338949850b070bed248c.jpg)

![abaa8a4d013a785bac18adc5d71c991c052154894d989387df49f704438a5dd7.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/abaa8a4d013a785bac18adc5d71c991c052154894d989387df49f704438a5dd7.jpg)

![bfcacfd88196fb65fe5e3ae060e4ebd4f732ded748c8542916a7602fe14ff1d6.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/bfcacfd88196fb65fe5e3ae060e4ebd4f732ded748c8542916a7602fe14ff1d6.jpg)

![c3ed8ee7efe51e7d8b45590bcd00cff4e5c75b9eccbeb699b47f7986523ebd46.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/c3ed8ee7efe51e7d8b45590bcd00cff4e5c75b9eccbeb699b47f7986523ebd46.jpg)

![cabbfbfdfc6b91068aa4df32e7b9fe6618cf930425f44fb27a7a3fb555d92bbf.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/cabbfbfdfc6b91068aa4df32e7b9fe6618cf930425f44fb27a7a3fb555d92bbf.jpg)

![de66e7c57f33d792226a856a9fb6f54ec6861224a0df363083b8c3f8b5f4fec5.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/images/de66e7c57f33d792226a856a9fb6f54ec6861224a0df363083b8c3f8b5f4fec5.jpg)

### Tables

![03a32ce7970e2fe561d04536bf8d12e09261883e152cfb720822b6bd3efe4d04.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/03a32ce7970e2fe561d04536bf8d12e09261883e152cfb720822b6bd3efe4d04.jpg)

![188f7bf2f572e1df931c67d40753b77d5422382bc55be79d1989e8af5d044f24.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/188f7bf2f572e1df931c67d40753b77d5422382bc55be79d1989e8af5d044f24.jpg)

![1a3d34938e10f8509391258e41e207fd5a036e8658e238bed9b3351015c01a3c.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/1a3d34938e10f8509391258e41e207fd5a036e8658e238bed9b3351015c01a3c.jpg)

![29278dbae0fdda115d533c7a2804beefd89afa93b780be56bb63be0f61596282.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/29278dbae0fdda115d533c7a2804beefd89afa93b780be56bb63be0f61596282.jpg)

![467a3d04bd4ba86f21a0fb14ce8a1a89895979c2686f7019f3bf245da5eabac9.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/467a3d04bd4ba86f21a0fb14ce8a1a89895979c2686f7019f3bf245da5eabac9.jpg)

![5f85d9d2a91d0b2e2b46bb09de24d2f9c1e239112792e78c5b57feff0179920d.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/5f85d9d2a91d0b2e2b46bb09de24d2f9c1e239112792e78c5b57feff0179920d.jpg)

![7b155c8777845431539a675bd695d5327ac88b4ec012afd27de093ff67420fba.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/7b155c8777845431539a675bd695d5327ac88b4ec012afd27de093ff67420fba.jpg)

![8c5a0714786019ee496851c608ae72eaa66cc9e8f749a48f8ca077faf3350543.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/8c5a0714786019ee496851c608ae72eaa66cc9e8f749a48f8ca077faf3350543.jpg)

![8d7b49fd6ed6c6721ab602984abf537d37bb4a5bf1e21a179d701e1845d50517.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/8d7b49fd6ed6c6721ab602984abf537d37bb4a5bf1e21a179d701e1845d50517.jpg)

![98349d559a5aa499503e44ff82a67cd7a5ca76b202449531950d704c8d3216fa.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/98349d559a5aa499503e44ff82a67cd7a5ca76b202449531950d704c8d3216fa.jpg)

![afe0a61fbdef14c63986aa7b6c4912dfa9b17913271785e6aca2c66cd0cc2b28.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/afe0a61fbdef14c63986aa7b6c4912dfa9b17913271785e6aca2c66cd0cc2b28.jpg)

![bb9cd3ac828a573baf625d557e028fa8e15b9b40044420894d71c5ff79e6ed1d.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/bb9cd3ac828a573baf625d557e028fa8e15b9b40044420894d71c5ff79e6ed1d.jpg)

![dc6e097986696c0e1dbbe1b7b632680da4d8866e92d24356ba5c5708e095dd8f.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/dc6e097986696c0e1dbbe1b7b632680da4d8866e92d24356ba5c5708e095dd8f.jpg)

![dce8ed63319d8f7b3d91693d1d8d1d6e37901cb078da5e129852833fbc36842b.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/dce8ed63319d8f7b3d91693d1d8d1d6e37901cb078da5e129852833fbc36842b.jpg)

![e298370d6228fd61e27bf0908fbf844661a6a83e8001fdfcd21b2cc2d7da353d.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/e298370d6228fd61e27bf0908fbf844661a6a83e8001fdfcd21b2cc2d7da353d.jpg)

![f3411688255020d482bb2ee438983aa4fbaa7245afed3f19b23dfe4372700ca0.jpg](../icml_results/6298_ACTIVE-o3%20_%20Empowering%20MLLMs%20with%20Active%20Perception%20via%20Pure%20Reinforcement%20Learning/tables/f3411688255020d482bb2ee438983aa4fbaa7245afed3f19b23dfe4372700ca0.jpg)

<a id="paper-6247"></a>

## ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response


### Images

![12290d75ad46850c289fab9a025631f05f6a185b0833f24f5d33a1f59d8f4a3a.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/12290d75ad46850c289fab9a025631f05f6a185b0833f24f5d33a1f59d8f4a3a.jpg)

![201d8a122f1dcae57629b9952e1b8f3c3206973bcbac1780c9aca214308cb7b2.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/201d8a122f1dcae57629b9952e1b8f3c3206973bcbac1780c9aca214308cb7b2.jpg)

![42dce74779db2d9a3c751416ff0af00e11a4dcbc919a9d973da01f9a9f42c517.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/42dce74779db2d9a3c751416ff0af00e11a4dcbc919a9d973da01f9a9f42c517.jpg)

![474d6ab342198cc55c4ca4cbfaa00b79cac39c89115bd2767f35db242d0fd779.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/474d6ab342198cc55c4ca4cbfaa00b79cac39c89115bd2767f35db242d0fd779.jpg)

![4a2db07521fb7ab7eec4b0d4e97539e85047023c7ea61694494e489ddf4d5b57.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/4a2db07521fb7ab7eec4b0d4e97539e85047023c7ea61694494e489ddf4d5b57.jpg)

![4f6209b86b2d0d4ce866eb35b85ae324fba0f9c193fe4be5013263f4cd22877e.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/4f6209b86b2d0d4ce866eb35b85ae324fba0f9c193fe4be5013263f4cd22877e.jpg)

![5fb288fbf28043b3b9d695bd6add1c0fe2c6c4812bfc19a31294f8a349955aca.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/5fb288fbf28043b3b9d695bd6add1c0fe2c6c4812bfc19a31294f8a349955aca.jpg)

![b3e949c9b838321fcdeaada594c24ee96e5685c2daae85b3c4bf27e6136cea42.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/b3e949c9b838321fcdeaada594c24ee96e5685c2daae85b3c4bf27e6136cea42.jpg)

![b9f02c516205402436aa5558ff2be2ef5045946d0d4e9b27e10bb9b483a545cb.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/b9f02c516205402436aa5558ff2be2ef5045946d0d4e9b27e10bb9b483a545cb.jpg)

![c81d0c8737058b5c24c082f4b3ed6834d0798bb010d05c68551667d1a5d3231f.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/c81d0c8737058b5c24c082f4b3ed6834d0798bb010d05c68551667d1a5d3231f.jpg)

![d06483875a5ac58d6eac9cf64b532d0f43ee45755d21c749814187f901622d6a.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/d06483875a5ac58d6eac9cf64b532d0f43ee45755d21c749814187f901622d6a.jpg)

![e2a3700c698a97465488253c5a45fadc00a73c513c64a0a75e329ca0b55c5ce2.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/e2a3700c698a97465488253c5a45fadc00a73c513c64a0a75e329ca0b55c5ce2.jpg)

![e90a39311d8398ae53310669ba197173943e78ea8ce1f4defb67cd603e7ba403.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/e90a39311d8398ae53310669ba197173943e78ea8ce1f4defb67cd603e7ba403.jpg)

![ec0d385fe418bb6c47b2ae796c4ca6af2030a5ca3417e6cfec833d1726b73a1d.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/ec0d385fe418bb6c47b2ae796c4ca6af2030a5ca3417e6cfec833d1726b73a1d.jpg)

![f0f7c2df79c02e242c2f49b8f0a05c2bbc42ca7d3b74d0ea9154fce25a9ffb0d.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/images/f0f7c2df79c02e242c2f49b8f0a05c2bbc42ca7d3b74d0ea9154fce25a9ffb0d.jpg)

### Tables

![064e44eaa510415d99647069aac597761d7eab7abb382da22479fca7f03cf1cc.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/064e44eaa510415d99647069aac597761d7eab7abb382da22479fca7f03cf1cc.jpg)

![0a10b8cf32cfb16910c5e72e439a749f9e71f3ac14d87716705b5e28110e7029.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/0a10b8cf32cfb16910c5e72e439a749f9e71f3ac14d87716705b5e28110e7029.jpg)

![45de96092a77c740b2a91c64b36901de773885cceb98c47cdf83013be507e829.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/45de96092a77c740b2a91c64b36901de773885cceb98c47cdf83013be507e829.jpg)

![568247870b6d7a4cc8076e6d71cc9098f24f5c34db12463936e135d4803f0b7a.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/568247870b6d7a4cc8076e6d71cc9098f24f5c34db12463936e135d4803f0b7a.jpg)

![71fd84c7960dc4eda2b5480ebec050295a6206f86c6702fc6fad93a33ff50ee1.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/71fd84c7960dc4eda2b5480ebec050295a6206f86c6702fc6fad93a33ff50ee1.jpg)

![74481f90636386e3405103a5a7f4022a6c39194a75299ac3bdc553a6639d6595.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/74481f90636386e3405103a5a7f4022a6c39194a75299ac3bdc553a6639d6595.jpg)

![8ca90dd863652d305ec9fbcaa9e5b45283c973f9152801d2347af7837922349f.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/8ca90dd863652d305ec9fbcaa9e5b45283c973f9152801d2347af7837922349f.jpg)

![97e70de872b4eeede9e2d4853869d700c346a7b67dbf6367361bff397bbd95b4.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/97e70de872b4eeede9e2d4853869d700c346a7b67dbf6367361bff397bbd95b4.jpg)

![b493e04c05f2232cbb3013d238a447099065bbdf94658c546f88e577e60f2a17.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/b493e04c05f2232cbb3013d238a447099065bbdf94658c546f88e577e60f2a17.jpg)

![c774a301bcc2e0898f933f1583968f0d1b940814fa3ea5b3f3cab405f6755180.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/c774a301bcc2e0898f933f1583968f0d1b940814fa3ea5b3f3cab405f6755180.jpg)

![ea1fc925cc905edb72c0e970b5dd869174f48e68d30bfc2e811094bba669d263.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/ea1fc925cc905edb72c0e970b5dd869174f48e68d30bfc2e811094bba669d263.jpg)

![f04898cdb7cbde078360f335d1c8a9900d84554485a70c54d4b2a5e910734e3f.jpg](../icml_results/6299_ProAct_%20A%20Benchmark%20and%20Multimodal%20Framework%20for%20Structure-Aware%20Proactive%20Response/tables/f04898cdb7cbde078360f335d1c8a9900d84554485a70c54d4b2a5e910734e3f.jpg)

<a id="paper-6248"></a>

## The Geometric Reasoner: Manifold-Informed Latent Foresight Search for Long-Context Reasoning


### Images

![008021aa126538fcb44f34ecee02f0ad7c835881424b76906e8a070ebf7ca273.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/008021aa126538fcb44f34ecee02f0ad7c835881424b76906e8a070ebf7ca273.jpg)

![03b8cf6df10747dc198fc6c5001cc3f01b1c716c193ccc58dd9635fb3b536df1.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/03b8cf6df10747dc198fc6c5001cc3f01b1c716c193ccc58dd9635fb3b536df1.jpg)

![06f81f1ac28904e5bfc5e0cd8ef9267088589c8c6d717139177911669a73e6d9.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/06f81f1ac28904e5bfc5e0cd8ef9267088589c8c6d717139177911669a73e6d9.jpg)

![0c98e3ee967c947f237bf9e72f84107888a379344f04dce9fe1078535d4aaeea.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/0c98e3ee967c947f237bf9e72f84107888a379344f04dce9fe1078535d4aaeea.jpg)

![18b93bbc2a3477d329d80098911dba2339adaede4f6f5c3e17fac59d92bf3de9.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/18b93bbc2a3477d329d80098911dba2339adaede4f6f5c3e17fac59d92bf3de9.jpg)

![26af90a5a756610e37a845d1bc0e4c762b36de4e55e3f4a7c0bcd241b3190563.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/26af90a5a756610e37a845d1bc0e4c762b36de4e55e3f4a7c0bcd241b3190563.jpg)

![63a413f7a82780be65662db79b6af555bdf9329b3f0c628172143c1788ec437a.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/63a413f7a82780be65662db79b6af555bdf9329b3f0c628172143c1788ec437a.jpg)

![68c8650f366bd20cc0495c0b97cd1532409c41e628501a4f174ed34d9f1c26a0.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/68c8650f366bd20cc0495c0b97cd1532409c41e628501a4f174ed34d9f1c26a0.jpg)

![93928ec0e341228c47e37499e9ee11cd97c24b72bdae545925d8c4467b3b2fc8.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/93928ec0e341228c47e37499e9ee11cd97c24b72bdae545925d8c4467b3b2fc8.jpg)

![9e8125ec234152866691ce21ff95aca67b1082814d4f67307ffe7afb22cf231a.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/9e8125ec234152866691ce21ff95aca67b1082814d4f67307ffe7afb22cf231a.jpg)

![c43b21485663be23b9c81bb72e2b9f3fbf7bed5ee84def152064e277d5c30df6.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/c43b21485663be23b9c81bb72e2b9f3fbf7bed5ee84def152064e277d5c30df6.jpg)

![c912cf5ead76a02f51f2b53d5ee8e18dd8a5fa3c92c3b1831673c5478b600f35.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/c912cf5ead76a02f51f2b53d5ee8e18dd8a5fa3c92c3b1831673c5478b600f35.jpg)

![ce144059f24635d3352206b58f3a8f3a80fdffb325518ce6de00f582fa92553d.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/images/ce144059f24635d3352206b58f3a8f3a80fdffb325518ce6de00f582fa92553d.jpg)

### Tables

![12037e043fb193a5892a87b0136d53aa9df15d159f546ac22478ab564e32124e.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/12037e043fb193a5892a87b0136d53aa9df15d159f546ac22478ab564e32124e.jpg)

![151fb107d296504ee114f2710949c8cb362cd53a06be0364e61f7a72f5b7842a.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/151fb107d296504ee114f2710949c8cb362cd53a06be0364e61f7a72f5b7842a.jpg)

![226b28e26536ca1d9e8b8c980cd2c653557c7ac329f3a8e58c112380d1da0e55.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/226b28e26536ca1d9e8b8c980cd2c653557c7ac329f3a8e58c112380d1da0e55.jpg)

![2cd8ce87d71ec866565c211b19f09ec4e044cb4bde383d7e07429f276b529af7.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/2cd8ce87d71ec866565c211b19f09ec4e044cb4bde383d7e07429f276b529af7.jpg)

![2cf0580c56a1613eb4e68df4dbbcb5d2a0474a74a1b70b9528f7a2909caef53a.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/2cf0580c56a1613eb4e68df4dbbcb5d2a0474a74a1b70b9528f7a2909caef53a.jpg)

![61f96711bb0d4121c24c4dd9e44722dca18423a351e275494f845616542d6900.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/61f96711bb0d4121c24c4dd9e44722dca18423a351e275494f845616542d6900.jpg)

![75a88e474ec3656005df3f806fc018a8c4bf34294056b4c9b47ff9e21f1aa375.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/75a88e474ec3656005df3f806fc018a8c4bf34294056b4c9b47ff9e21f1aa375.jpg)

![76e335fe9932c1b603f433c17805887adde361827061f5258a6539e031dbbdf0.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/76e335fe9932c1b603f433c17805887adde361827061f5258a6539e031dbbdf0.jpg)

![9ed8f8ae02d0e663917b555b8dfd354b1b9cb7b14dce138f479babead21d294c.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/9ed8f8ae02d0e663917b555b8dfd354b1b9cb7b14dce138f479babead21d294c.jpg)

![bcacb675f3452701840ebf38aed0f3aac56e3658be0d5d4b1e558fe143e6b283.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/bcacb675f3452701840ebf38aed0f3aac56e3658be0d5d4b1e558fe143e6b283.jpg)

![caec1ef0b00c0c920e55557864d2c18fc58b3fcf02f62880c92c78f0bb14ee41.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/caec1ef0b00c0c920e55557864d2c18fc58b3fcf02f62880c92c78f0bb14ee41.jpg)

![d3d8c1d91ef3e7e9c653c0645751e79efc82d1167d3a45a92a59d1f23e2e49c2.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/d3d8c1d91ef3e7e9c653c0645751e79efc82d1167d3a45a92a59d1f23e2e49c2.jpg)

![e314f3835f1a77b6fde8f6933ea106bf849d1835026141bbb87379711a988a18.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/e314f3835f1a77b6fde8f6933ea106bf849d1835026141bbb87379711a988a18.jpg)

![fc307a2ada61ac1cb6cd56fc66ad054bb8467531907e84b09913be60e5bad96e.jpg](../icml_results/6300_The%20Geometric%20Reasoner_%20Manifold-Informed%20Latent%20Foresight%20Search%20for%20Long-Context%20Reasoning/tables/fc307a2ada61ac1cb6cd56fc66ad054bb8467531907e84b09913be60e5bad96e.jpg)

<a id="paper-6249"></a>

## Geometric Control of Out-of-Distribution Shift in Safe Offline RL


### Images

![19e04fa54cfaeb110554641464bfb404f2aee0c94a2718d4365bf76203613a72.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/19e04fa54cfaeb110554641464bfb404f2aee0c94a2718d4365bf76203613a72.jpg)

![342d699dc7de912aa89608eafb0400fbbd9ba7635bfe9aa22f2d0e965c90b6a9.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/342d699dc7de912aa89608eafb0400fbbd9ba7635bfe9aa22f2d0e965c90b6a9.jpg)

![47715fe3e8b1a93b588f8814b2562632745ea88d792e035ec6bdb26da006f6e4.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/47715fe3e8b1a93b588f8814b2562632745ea88d792e035ec6bdb26da006f6e4.jpg)

![6217fcacb00513039f6bb6d76a50651bf3df38162d08dec0b79060bff719c5be.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/6217fcacb00513039f6bb6d76a50651bf3df38162d08dec0b79060bff719c5be.jpg)

![8515d26564946738953ae9c183c1a6c22ec26050d47fa90e3f996498d9ce2085.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/8515d26564946738953ae9c183c1a6c22ec26050d47fa90e3f996498d9ce2085.jpg)

![8dcee37295baf129f2d4c85428c110dbb87845fc7d24de52ddff88bd9fb6dec5.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/8dcee37295baf129f2d4c85428c110dbb87845fc7d24de52ddff88bd9fb6dec5.jpg)

![fa232e8ff7c0bc22265a4e594e0a23c433b33f602bfbf64505a1877a9f1af95b.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/images/fa232e8ff7c0bc22265a4e594e0a23c433b33f602bfbf64505a1877a9f1af95b.jpg)

### Tables

![015d85ec9223fbba5f01b9fe2276bc870f3a7181e26b2d9e5e405ef95e424cdb.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/015d85ec9223fbba5f01b9fe2276bc870f3a7181e26b2d9e5e405ef95e424cdb.jpg)

![3122c2150cf82d5a5f41fe99fa234d03c79396fd5721627d986d80078eb5fef5.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/3122c2150cf82d5a5f41fe99fa234d03c79396fd5721627d986d80078eb5fef5.jpg)

![486395f6fb579ebf3f84c93d2c82c1c44e1079739fffbfee821052cf36b5779b.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/486395f6fb579ebf3f84c93d2c82c1c44e1079739fffbfee821052cf36b5779b.jpg)

![48b44ba9568ef41ee280f823a8960017442fda6b219756093b9b794c28acb5ab.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/48b44ba9568ef41ee280f823a8960017442fda6b219756093b9b794c28acb5ab.jpg)

![4c16a19c88c807070154b470861d67072cb3e65c0967c142d21633c116a4fdbe.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/4c16a19c88c807070154b470861d67072cb3e65c0967c142d21633c116a4fdbe.jpg)

![5178ab221db74fab95d69f9a6916c92a7a43f011a66b3801c166c4a9161cc49e.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/5178ab221db74fab95d69f9a6916c92a7a43f011a66b3801c166c4a9161cc49e.jpg)

![56921322f855665889db312060e4f49b413a9c318ea977d0873715ffe7bdf824.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/56921322f855665889db312060e4f49b413a9c318ea977d0873715ffe7bdf824.jpg)

![99d6f8599e3a43fc77517a4cf1cc19dca9525e4ce056fff0680327dddcab1566.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/99d6f8599e3a43fc77517a4cf1cc19dca9525e4ce056fff0680327dddcab1566.jpg)

![ca17f62673eddc61d8f34bf4cba8121cf1bf8bbe728cad0470ec22498fef78dc.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/ca17f62673eddc61d8f34bf4cba8121cf1bf8bbe728cad0470ec22498fef78dc.jpg)

![d12f466586d1f77779bcb12eded2f4c893329e1065db73d430c5602e33995ae1.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/d12f466586d1f77779bcb12eded2f4c893329e1065db73d430c5602e33995ae1.jpg)

![f926cb4a341c90157f9053c42da388b2034e0c7cb5e4150515920aca3bef3ff2.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/f926cb4a341c90157f9053c42da388b2034e0c7cb5e4150515920aca3bef3ff2.jpg)

![f9b7f469c008c8a68bee7221b0e5afb0d236a6a67558c460f6e04ecca75d6d79.jpg](../icml_results/6301_Geometric%20Control%20of%20Out-of-Distribution%20Shift%20in%20Safe%20Offline%20RL/tables/f9b7f469c008c8a68bee7221b0e5afb0d236a6a67558c460f6e04ecca75d6d79.jpg)

<a id="paper-6250"></a>

## Mitigating Manifold Departure: Uncertainty-Aware Subspace Rectification for Trustworthy MLLM Decoding


### Images

![011dac3487ab7755db8002d7781cda5ec5ee30e26d9bd08199cb979409e249bd.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/images/011dac3487ab7755db8002d7781cda5ec5ee30e26d9bd08199cb979409e249bd.jpg)

![7e1d5b7b8a4dcc19fdd78dba66e8dd3335fa569a03d4052dffc19893761adbd9.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/images/7e1d5b7b8a4dcc19fdd78dba66e8dd3335fa569a03d4052dffc19893761adbd9.jpg)

![9bd9339c1a798399ce5a5dcf2a9819b8d67fb49e16861abaecbba123555d9955.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/images/9bd9339c1a798399ce5a5dcf2a9819b8d67fb49e16861abaecbba123555d9955.jpg)

![b07957cf9ddac9286676aefacf800500a714260dba34584ba756fed8120acdbc.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/images/b07957cf9ddac9286676aefacf800500a714260dba34584ba756fed8120acdbc.jpg)

![b89457577eea2aa39e04b463d1db6f2d809ce94be765b12f125353b3bb07bcbc.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/images/b89457577eea2aa39e04b463d1db6f2d809ce94be765b12f125353b3bb07bcbc.jpg)

![d38fe1a2b011b926614f6f3f1aea3b81b51f4758f83442681e1d8f9bccff2f4c.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/images/d38fe1a2b011b926614f6f3f1aea3b81b51f4758f83442681e1d8f9bccff2f4c.jpg)

### Tables

![231b1fbd5b020e8508c4cfc33db34639b1459c9a67994bb575d2dc8306379f2b.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/231b1fbd5b020e8508c4cfc33db34639b1459c9a67994bb575d2dc8306379f2b.jpg)

![243d068d58d88044c39e29438eefeae9569c89edf09bdb20b9bebf3fe2268abb.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/243d068d58d88044c39e29438eefeae9569c89edf09bdb20b9bebf3fe2268abb.jpg)

![2736395aa9afa2f81da5b66802ecda5564c922814919e192a397945beda36323.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/2736395aa9afa2f81da5b66802ecda5564c922814919e192a397945beda36323.jpg)

![2ddeb25240e58b70db5b7cb9f7bd4db6701f3d896317e01403a7c871aeb2af5e.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/2ddeb25240e58b70db5b7cb9f7bd4db6701f3d896317e01403a7c871aeb2af5e.jpg)

![30c1a657a56b0cb7c4bc9a496815021836caf5ef32c02d7e577d829e401c5ecf.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/30c1a657a56b0cb7c4bc9a496815021836caf5ef32c02d7e577d829e401c5ecf.jpg)

![4121958e3e86ff19cf8e147c8f68354f507abc4689e0b1be11faf691a71da650.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/4121958e3e86ff19cf8e147c8f68354f507abc4689e0b1be11faf691a71da650.jpg)

![accb74eac38d6db067e2bc36b60cc19bc0730ac04ee8a54558783b39382a959b.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/accb74eac38d6db067e2bc36b60cc19bc0730ac04ee8a54558783b39382a959b.jpg)

![c6f04a198727c7bc92531b008ce432f426469374ebcb998bd71177b985ccb5ea.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/c6f04a198727c7bc92531b008ce432f426469374ebcb998bd71177b985ccb5ea.jpg)

![d1b41438f90174d89f82ce075071efa4e2e8bdbc21223b2696085e4528a68814.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/d1b41438f90174d89f82ce075071efa4e2e8bdbc21223b2696085e4528a68814.jpg)

![e63b1753da441726aefd213830fdfb375341232c0d9fbb2b262cabae9b081354.jpg](../icml_results/6302_Mitigating%20Manifold%20Departure_%20Uncertainty-Aware%20Subspace%20Rectification%20for%20Trustworthy%20MLLM%20Decodin/tables/e63b1753da441726aefd213830fdfb375341232c0d9fbb2b262cabae9b081354.jpg)

<a id="paper-6251"></a>

## Efficient Adaptive Testing via Gradient Path Matching Subset Selection for AI Education


### Images

![4972c52cdc8289a306d98a2a87a2a7b729e85db773cce6c6a713908ad7302e6f.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/images/4972c52cdc8289a306d98a2a87a2a7b729e85db773cce6c6a713908ad7302e6f.jpg)

![799a34f4dbb1a77426519a94ac9a682ba6da775ea7f781ca5d0ebcb65506364c.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/images/799a34f4dbb1a77426519a94ac9a682ba6da775ea7f781ca5d0ebcb65506364c.jpg)

![7c44334cf26514ff7a0e6df7779ca52c588e628c20b16b81fc9be29b8458e1c3.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/images/7c44334cf26514ff7a0e6df7779ca52c588e628c20b16b81fc9be29b8458e1c3.jpg)

![ea4d02288acb17d4fe71b2146aba58601dfa7d0d6f150a6c08d97cd862c84fcf.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/images/ea4d02288acb17d4fe71b2146aba58601dfa7d0d6f150a6c08d97cd862c84fcf.jpg)

![f2d9c929eaa7ca2677c9f59cb34bebc95bc801e54e507fe8cd01cf0f7bdcda67.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/images/f2d9c929eaa7ca2677c9f59cb34bebc95bc801e54e507fe8cd01cf0f7bdcda67.jpg)

### Tables

![171fc08b1fecc1ee86cbe1e5619d33b97b0364f753aab8cc9657e38ec919244a.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/tables/171fc08b1fecc1ee86cbe1e5619d33b97b0364f753aab8cc9657e38ec919244a.jpg)

![1b81a20def68184e3e67e2079881ee8c51bd32596cbbdf71687b0425dc0dcc41.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/tables/1b81a20def68184e3e67e2079881ee8c51bd32596cbbdf71687b0425dc0dcc41.jpg)

![2f9bf832301dcc63243e8f5c343d5bb07fd0d0db2db50653e3be2821abffd2f8.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/tables/2f9bf832301dcc63243e8f5c343d5bb07fd0d0db2db50653e3be2821abffd2f8.jpg)

![6a217955eeab62dee2f72e8feabb5cb17e27bd22d11de3f5aa9f885a01b5f41a.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/tables/6a217955eeab62dee2f72e8feabb5cb17e27bd22d11de3f5aa9f885a01b5f41a.jpg)

![8691e61d019f3274080cfc4325dd1b7bfb027f810190f32e890ed0df38d089ef.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/tables/8691e61d019f3274080cfc4325dd1b7bfb027f810190f32e890ed0df38d089ef.jpg)

![ea81177fac50be3ff508f42300df5819a1cbb507794de79e39e00302e5c27653.jpg](../icml_results/6303_Efficient%20Adaptive%20Testing%20via%20Gradient%20Path%20Matching%20Subset%20Selection%20for%20AI%20Education/tables/ea81177fac50be3ff508f42300df5819a1cbb507794de79e39e00302e5c27653.jpg)

<a id="paper-6252"></a>

## Rethinking Graph Transformers as Graph Signal Denoisers: The Role of Block-Diagonal Priors


### Images

![09d4e6bbaa5a7ed4d409ea925a7bdf0c8f13463738f70edd09dfac9678b718d2.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/09d4e6bbaa5a7ed4d409ea925a7bdf0c8f13463738f70edd09dfac9678b718d2.jpg)

![1643faf36f43175b958fd8cca7f394ce5667b21542f9cac77005f1cb2ab4a0f8.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/1643faf36f43175b958fd8cca7f394ce5667b21542f9cac77005f1cb2ab4a0f8.jpg)

![2162849ca99a4292fb9d65ea5e3e9c95c3d84382291122e5e980bbd9045baab9.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/2162849ca99a4292fb9d65ea5e3e9c95c3d84382291122e5e980bbd9045baab9.jpg)

![3c4a3e567e952d1433f1b5235e1d328df05e00aabd5a5c879a43e522e147cd75.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/3c4a3e567e952d1433f1b5235e1d328df05e00aabd5a5c879a43e522e147cd75.jpg)

![4af5a60720d1c31f09177072249e7bce3c9ea04626d7da6b2bc8b4f0cc1d28fe.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/4af5a60720d1c31f09177072249e7bce3c9ea04626d7da6b2bc8b4f0cc1d28fe.jpg)

![ab0d4bb5347a3f6040c57574bdb48cd2e52f3cac32043d1bc39e38c604dd756d.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/ab0d4bb5347a3f6040c57574bdb48cd2e52f3cac32043d1bc39e38c604dd756d.jpg)

![bb51bf988dee6e80a9ad0899a5c30741d710424f3c3d7524aae8a5240494f32b.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/bb51bf988dee6e80a9ad0899a5c30741d710424f3c3d7524aae8a5240494f32b.jpg)

![ebb06dcdc3278c5c907df9762617a7a0dd1177c146b0e16798f015fd8a32b1dd.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/images/ebb06dcdc3278c5c907df9762617a7a0dd1177c146b0e16798f015fd8a32b1dd.jpg)

### Tables

![0850aa4b350f4e8116bb23f956b399d9fd68820304b81b5b38d2c862a263d09e.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/0850aa4b350f4e8116bb23f956b399d9fd68820304b81b5b38d2c862a263d09e.jpg)

![18a979c9fbce8d4c725b11d6af3bb534f02e40815dbbbd0da5123aa6d80b87a6.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/18a979c9fbce8d4c725b11d6af3bb534f02e40815dbbbd0da5123aa6d80b87a6.jpg)

![82ef76be44eecb9d3c39ab809d7642c729c9ca4c613e756679b6b7399cb19b51.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/82ef76be44eecb9d3c39ab809d7642c729c9ca4c613e756679b6b7399cb19b51.jpg)

![82f4f542797413855457872b024bbc69d30c6f6db999b6502f73cf85c104be33.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/82f4f542797413855457872b024bbc69d30c6f6db999b6502f73cf85c104be33.jpg)

![af8f9fd7b5a8be376fdac469376ed3dfb38c3e5b6805c1431fc2197fe4d6ee2d.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/af8f9fd7b5a8be376fdac469376ed3dfb38c3e5b6805c1431fc2197fe4d6ee2d.jpg)

![b015cc415a67b7285a8235e67b7485d1c1fec4f834125d74b9cffaa8d01a3320.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/b015cc415a67b7285a8235e67b7485d1c1fec4f834125d74b9cffaa8d01a3320.jpg)

![d478290e784f67582834b03969440cdb98864ff2befac4faf398e22f9012e42e.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/d478290e784f67582834b03969440cdb98864ff2befac4faf398e22f9012e42e.jpg)

![f3513809ed9b3a709a6ec0843ca58f683b28a822180c9ae394178d215849bcba.jpg](../icml_results/6304_Rethinking%20Graph%20Transformers%20as%20Graph%20Signal%20Denoisers_%20The%20Role%20of%20Block-Diagonal%20Priors/tables/f3513809ed9b3a709a6ec0843ca58f683b28a822180c9ae394178d215849bcba.jpg)

<a id="paper-6253"></a>

## Understanding the Ability of LLMs to Handle Character-Level Perturbation


### Images

![31a9fe07ba194644d3fb35888e56a74ec3296a07c0f4dea65cbd1b455e7a9bb3.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/31a9fe07ba194644d3fb35888e56a74ec3296a07c0f4dea65cbd1b455e7a9bb3.jpg)

![3319e4d70a5ae261636dc54018978b295e5e09f314ee933710de82a6c296e61f.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/3319e4d70a5ae261636dc54018978b295e5e09f314ee933710de82a6c296e61f.jpg)

![67712d6f5cd081145cf944daba755db91a52db530a890e067e642a70621c7250.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/67712d6f5cd081145cf944daba755db91a52db530a890e067e642a70621c7250.jpg)

![7f492ff6b1a77b1a52423b0d86667f708c08cfd1dfe8b16ffaf9307f692d917d.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/7f492ff6b1a77b1a52423b0d86667f708c08cfd1dfe8b16ffaf9307f692d917d.jpg)

![8a08c243e34555f525b2cf7225cf30207b7fcbf41bb35046e0d6f94ebc2b19d4.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/8a08c243e34555f525b2cf7225cf30207b7fcbf41bb35046e0d6f94ebc2b19d4.jpg)

![8a19079843bec348545b06c15235c7a6e191b28d3f11fd2928338f53abed2f82.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/8a19079843bec348545b06c15235c7a6e191b28d3f11fd2928338f53abed2f82.jpg)

![8be74b51241775dbf1d1889739119bbfc786ea28fbeae524624b0e28bd715f25.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/8be74b51241775dbf1d1889739119bbfc786ea28fbeae524624b0e28bd715f25.jpg)

![c0c1c85233bb1ad53be5002625e5ba9c47ec2b04480211fffe2b77f5b3c47423.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/c0c1c85233bb1ad53be5002625e5ba9c47ec2b04480211fffe2b77f5b3c47423.jpg)

![c79ad5bfb4c42069f8fddbfb5244a1dfe31f632b519e6db3d8d1eb92948111ec.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/c79ad5bfb4c42069f8fddbfb5244a1dfe31f632b519e6db3d8d1eb92948111ec.jpg)

![d7f9c4b221fe5466f3c8227d7fe36816cb686470b58b7dcb71ca33bb014c7008.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/d7f9c4b221fe5466f3c8227d7fe36816cb686470b58b7dcb71ca33bb014c7008.jpg)

![fee69b0d8e9b7d78c8b9536d896b460b0961925536ca0026888ca4f632aa1f6e.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/images/fee69b0d8e9b7d78c8b9536d896b460b0961925536ca0026888ca4f632aa1f6e.jpg)

### Tables

![10111b558332ccea99ba428f1e0473163934d41982a66260efdc40ca0a61e7cd.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/10111b558332ccea99ba428f1e0473163934d41982a66260efdc40ca0a61e7cd.jpg)

![126d667cdb8f5ae7db428868a3b4bc5b00ce2548ecbe58c86e40927a4f4db9ec.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/126d667cdb8f5ae7db428868a3b4bc5b00ce2548ecbe58c86e40927a4f4db9ec.jpg)

![30c1f2c923baf181754b783c2b638db1cde9ce0f196e86785b1dddd9cb2f820a.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/30c1f2c923baf181754b783c2b638db1cde9ce0f196e86785b1dddd9cb2f820a.jpg)

![38cf83cb2a85c8ac034f021e1b9ae92848bc4a6b2dbf25bd6f15fe21f5c196e1.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/38cf83cb2a85c8ac034f021e1b9ae92848bc4a6b2dbf25bd6f15fe21f5c196e1.jpg)

![569e0f0b19722ce117653bca1863d3dfcb8509be6c214460f4df5170d7d4aa2c.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/569e0f0b19722ce117653bca1863d3dfcb8509be6c214460f4df5170d7d4aa2c.jpg)

![5a1a330fc2d2342245e332dc302459374f39f8e50494bfb9959458ab8c432340.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/5a1a330fc2d2342245e332dc302459374f39f8e50494bfb9959458ab8c432340.jpg)

![611da0ef49389175efb3926eb3c8b2b688d142b655efe916952fc1f1794f1f4b.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/611da0ef49389175efb3926eb3c8b2b688d142b655efe916952fc1f1794f1f4b.jpg)

![623414bd1e0f78072b445dd2ee75e526d0896461de7a575b9b1b09d9cef3fc8c.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/623414bd1e0f78072b445dd2ee75e526d0896461de7a575b9b1b09d9cef3fc8c.jpg)

![690ecd3aa4a452041802feeae85f23b3697272b3c639c9d92cd74129814485b3.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/690ecd3aa4a452041802feeae85f23b3697272b3c639c9d92cd74129814485b3.jpg)

![8584e455e8ffa7658165a25b257b632ada02df047736d8b6ad15bdc20ec8c6da.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/8584e455e8ffa7658165a25b257b632ada02df047736d8b6ad15bdc20ec8c6da.jpg)

![864522cfd22b56241eea25e5ebada96429ce74aa28253438b22bc65c5295797b.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/864522cfd22b56241eea25e5ebada96429ce74aa28253438b22bc65c5295797b.jpg)

![ab7b6ece1c8ca784b31d9a96ec23f4867f44eb681952d87109a2f83596d3f179.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/ab7b6ece1c8ca784b31d9a96ec23f4867f44eb681952d87109a2f83596d3f179.jpg)

![b9ff7c6a06bde8a33be27868a83648fdbb9877196fb6b08cbf57dc7f905488ed.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/b9ff7c6a06bde8a33be27868a83648fdbb9877196fb6b08cbf57dc7f905488ed.jpg)

![badc4975fc1c9c3929ba0ce541c1b233ab983b9b8df1181ac95216dc9298faf2.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/badc4975fc1c9c3929ba0ce541c1b233ab983b9b8df1181ac95216dc9298faf2.jpg)

![be4bc8e0a8a994eab47c0f0b02560f62e828e31595556b2ff134d0ff57cc3335.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/be4bc8e0a8a994eab47c0f0b02560f62e828e31595556b2ff134d0ff57cc3335.jpg)

![c3c031d7fef710881bcba52a9a1136c56ee7af87abfc7de0ac9f3fe2c94bc39e.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/c3c031d7fef710881bcba52a9a1136c56ee7af87abfc7de0ac9f3fe2c94bc39e.jpg)

![e7c072c7804ec79729d6ecbe2c4e19ad439b3d91144453057263feb91e57f7c5.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/e7c072c7804ec79729d6ecbe2c4e19ad439b3d91144453057263feb91e57f7c5.jpg)

![f72ea37c8570157070427f96ade031f2e99c577c1f9783205f01fc9c6d0cd75d.jpg](../icml_results/6305_Understanding%20the%20Ability%20of%20LLMs%20to%20Handle%20Character-Level%20Perturbation/tables/f72ea37c8570157070427f96ade031f2e99c577c1f9783205f01fc9c6d0cd75d.jpg)

<a id="paper-6254"></a>

## Training Language Model Agents to Find Vulnerabilities with CTF-Dojo


### Images

![475200ddd8125429a838672e5136198290ce52f677a645e8100f100667a876fa.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/images/475200ddd8125429a838672e5136198290ce52f677a645e8100f100667a876fa.jpg)

![4c68f95a1eb0ee46c794ac05878fe0c4a7e81aa111388b484ebf320af66a441c.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/images/4c68f95a1eb0ee46c794ac05878fe0c4a7e81aa111388b484ebf320af66a441c.jpg)

![502620a2836ec1de42c03602cf4da2b773d99ddc1d3570086691385b3cbf8aeb.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/images/502620a2836ec1de42c03602cf4da2b773d99ddc1d3570086691385b3cbf8aeb.jpg)

![5510de6e916af05daae36cb4b11833541a62489e23711f08eb1f00c1eb21c99b.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/images/5510de6e916af05daae36cb4b11833541a62489e23711f08eb1f00c1eb21c99b.jpg)

![6069acda2b205efcfae061cde3107b6bd8fa07a303e1c0494ca3b63b244e9b39.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/images/6069acda2b205efcfae061cde3107b6bd8fa07a303e1c0494ca3b63b244e9b39.jpg)

### Tables

![01df4ee0c465fe5ec5f359bc47a43f9b7dfd74b3aef8f22796f17f7cd8195a6f.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/01df4ee0c465fe5ec5f359bc47a43f9b7dfd74b3aef8f22796f17f7cd8195a6f.jpg)

![0dd2c31c8c4e6b31353e94af3f87c41d809aa12bead8ea49e51ac58b5b48cbd0.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/0dd2c31c8c4e6b31353e94af3f87c41d809aa12bead8ea49e51ac58b5b48cbd0.jpg)

![29e5cd4d7aa840b388157fc5f7c5e5868351136623ddd6fc7530626ac4556a0f.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/29e5cd4d7aa840b388157fc5f7c5e5868351136623ddd6fc7530626ac4556a0f.jpg)

![599c3a5c2540534d14ca6c1ce64563219bb80a6eb3b1972e3e5b088905755b03.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/599c3a5c2540534d14ca6c1ce64563219bb80a6eb3b1972e3e5b088905755b03.jpg)

![6c44533fec77d9477eb79dc6d31f2297e635e643b6f7def3911ee2f7b9aad8a9.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/6c44533fec77d9477eb79dc6d31f2297e635e643b6f7def3911ee2f7b9aad8a9.jpg)

![7e8e67e9e42617b39f23d47296f3785f36aec4157ea19d1c1cbb1dc7b5c7eebb.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/7e8e67e9e42617b39f23d47296f3785f36aec4157ea19d1c1cbb1dc7b5c7eebb.jpg)

![8899eb2d6d024d11f790c96f18fad99ac5c70f51b44ff3675eacbbb5cf28f52e.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/8899eb2d6d024d11f790c96f18fad99ac5c70f51b44ff3675eacbbb5cf28f52e.jpg)

![9267f361a59e0fff1cf9fae420150886be47a4c653dc0130c2b24a9647bd6338.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/9267f361a59e0fff1cf9fae420150886be47a4c653dc0130c2b24a9647bd6338.jpg)

![a87c5d9c720e3422853cc4c5e70400e0ec56a406c7e31509ad71c9619472bbcf.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/a87c5d9c720e3422853cc4c5e70400e0ec56a406c7e31509ad71c9619472bbcf.jpg)

![aa209a6c711517519d97bfacfdee73b6189b12136cced0e059d40b9b22dcc6a7.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/aa209a6c711517519d97bfacfdee73b6189b12136cced0e059d40b9b22dcc6a7.jpg)

![ba47ac1726cc6b0c46ce65a78b525051d9415530ea144c69062a9ea5eec1d706.jpg](../icml_results/6306_Training%20Language%20Model%20Agents%20to%20Find%20Vulnerabilities%20with%20CTF-Dojo/tables/ba47ac1726cc6b0c46ce65a78b525051d9415530ea144c69062a9ea5eec1d706.jpg)

<a id="paper-6255"></a>

## On the Spectral Unreachability of Brain Graph Learning


### Images

![01124e44b9661526093430ded0650fd3df77d9ca4ca9285534506f6cc29c16a6.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/01124e44b9661526093430ded0650fd3df77d9ca4ca9285534506f6cc29c16a6.jpg)

![2b38641c5ae16698ce1e86638d1a5f4efa69492206ec74652761ee3386dc12cc.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/2b38641c5ae16698ce1e86638d1a5f4efa69492206ec74652761ee3386dc12cc.jpg)

![adee3cf12c5a72c594a6ef9d24b7c95c12de844c1a70dd8e4b0032b663859479.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/adee3cf12c5a72c594a6ef9d24b7c95c12de844c1a70dd8e4b0032b663859479.jpg)

![bd0edfedace7ca1cba2e6018b4e3d2c95a27566f01beb3b3b6acc5eb6199c5b2.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/bd0edfedace7ca1cba2e6018b4e3d2c95a27566f01beb3b3b6acc5eb6199c5b2.jpg)

![d6f377eaa5bdbdd9f1946e98d395699b60cd560b56590c270589071a44800274.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/d6f377eaa5bdbdd9f1946e98d395699b60cd560b56590c270589071a44800274.jpg)

![f004e24846e0353cd070b34a955cd6f0b5d6c5bbac3144c1ec16383ed992b06a.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/f004e24846e0353cd070b34a955cd6f0b5d6c5bbac3144c1ec16383ed992b06a.jpg)

![f41a23145d021be95a113b8dc576b65afb0c028e64ff5e62c03e34354e97105a.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/images/f41a23145d021be95a113b8dc576b65afb0c028e64ff5e62c03e34354e97105a.jpg)

### Tables

![70d1e5e4184cba1dec900bd7d2985f5c1b5cf298e8474ed8163eccaf004e8625.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/tables/70d1e5e4184cba1dec900bd7d2985f5c1b5cf298e8474ed8163eccaf004e8625.jpg)

![8d9c47dfeff72ee17ff0145ceda0cf51e3bc3ef776e38e9591b43896f49f03be.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/tables/8d9c47dfeff72ee17ff0145ceda0cf51e3bc3ef776e38e9591b43896f49f03be.jpg)

![a57d39d2deef879642f00221713a264626bf463e0ab2f33bc6d3d8fc1f99768b.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/tables/a57d39d2deef879642f00221713a264626bf463e0ab2f33bc6d3d8fc1f99768b.jpg)

![a93d18b85b7c23cd7d1971f097d28f0b0d1557c5df2d41585463d4e6d63c52e5.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/tables/a93d18b85b7c23cd7d1971f097d28f0b0d1557c5df2d41585463d4e6d63c52e5.jpg)

![ba2f42a407e237e718f66a2a528b79992e5ab1ae320553205e42430a13277806.jpg](../icml_results/6307_On%20the%20Spectral%20Unreachability%20of%20Brain%20Graph%20Learning/tables/ba2f42a407e237e718f66a2a528b79992e5ab1ae320553205e42430a13277806.jpg)

<a id="paper-6256"></a>

## Phase-Type Variational Autoencoders for Heavy-Tailed Data


### Images

![41763b3e4f193081713aefde51030a11df6f428f4135bd670d9f25db44883341.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/41763b3e4f193081713aefde51030a11df6f428f4135bd670d9f25db44883341.jpg)

![465a00febd5f75dcdcad83b47a7e358db9ad8e11dccbe78075b9301d7acf7eb7.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/465a00febd5f75dcdcad83b47a7e358db9ad8e11dccbe78075b9301d7acf7eb7.jpg)

![64c421131510db0f4d1070fc2aedc30dea719f4f624c4ce10e4e241c71465878.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/64c421131510db0f4d1070fc2aedc30dea719f4f624c4ce10e4e241c71465878.jpg)

![8411defe3e8f35f18de62fa192d3d2816170f003ab8c0e32860cc86680f151ec.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/8411defe3e8f35f18de62fa192d3d2816170f003ab8c0e32860cc86680f151ec.jpg)

![dd30d0ac22ea73569db68de17b66fa99104b9f98b18651300d4ee556526c059f.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/dd30d0ac22ea73569db68de17b66fa99104b9f98b18651300d4ee556526c059f.jpg)

![ea0bfaac860069e868d090ac7ec4034c6b48b7da8d9fa127bcc630dafb6353b1.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/ea0bfaac860069e868d090ac7ec4034c6b48b7da8d9fa127bcc630dafb6353b1.jpg)

![eeec94b06dcc0a9166321c53eff3b227866c79d0b9939de7a89ec95a5b4c9382.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/eeec94b06dcc0a9166321c53eff3b227866c79d0b9939de7a89ec95a5b4c9382.jpg)

![f1306f29f473201e6de8c3ef505dca9c19a9b7d179409b5c4378d9c536e8bb4c.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/f1306f29f473201e6de8c3ef505dca9c19a9b7d179409b5c4378d9c536e8bb4c.jpg)

![f91e5a2a798a3d5e025e39582d3412c862da25a6cc4dd475573018a5b1a50683.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/images/f91e5a2a798a3d5e025e39582d3412c862da25a6cc4dd475573018a5b1a50683.jpg)

### Tables

![1ad0072db3cff0b82f8da33f4d34b661c03f561bfe51e1c899da02e0a84f9c61.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/tables/1ad0072db3cff0b82f8da33f4d34b661c03f561bfe51e1c899da02e0a84f9c61.jpg)

![546185e59edb3b60886ab939fc7e9edcee8f1f3d6be843f7d2983fa678d99ebf.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/tables/546185e59edb3b60886ab939fc7e9edcee8f1f3d6be843f7d2983fa678d99ebf.jpg)

![69de3b830198fd9de88f58facaa268246c5fa711bb7e442b48a33ec30281bf0e.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/tables/69de3b830198fd9de88f58facaa268246c5fa711bb7e442b48a33ec30281bf0e.jpg)

![87383b2db18d4570c9f2d6527f9a52364e4ebb1e4bbdb6690218cdca7db0c0b7.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/tables/87383b2db18d4570c9f2d6527f9a52364e4ebb1e4bbdb6690218cdca7db0c0b7.jpg)

![9c8a24a2eaf033d7142c725657cb5948009ccf9d4b5d0d84586346dcc6f8ca90.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/tables/9c8a24a2eaf033d7142c725657cb5948009ccf9d4b5d0d84586346dcc6f8ca90.jpg)

![af14770e60097e75529cf5638853b5b1ae2be61563d8b1997c0a9aa494d9bc00.jpg](../icml_results/6308_Phase-Type%20Variational%20Autoencoders%20for%20Heavy-Tailed%20Data/tables/af14770e60097e75529cf5638853b5b1ae2be61563d8b1997c0a9aa494d9bc00.jpg)

<a id="paper-6257"></a>

## Affine-Equivariant Kernel Space Encoding for NeRF Editing


### Images

![02f1bdfa23ce171d0596478cc3cff7e1ca3f344449bcc845d7349f8417e91db4.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/02f1bdfa23ce171d0596478cc3cff7e1ca3f344449bcc845d7349f8417e91db4.jpg)

![28d7978c8f3661b8a6add5a320a1a4e3397a750c74b13d3d4558aca126d71724.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/28d7978c8f3661b8a6add5a320a1a4e3397a750c74b13d3d4558aca126d71724.jpg)

![472b36029705eb69ceed2fd5259792eb1102d8473ae5db485fb129538ad7fc4a.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/472b36029705eb69ceed2fd5259792eb1102d8473ae5db485fb129538ad7fc4a.jpg)

![47ecf58a1e109e18f23402cad220e78da2a278825da8cac86feea8a24248b615.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/47ecf58a1e109e18f23402cad220e78da2a278825da8cac86feea8a24248b615.jpg)

![4e52a3bf8b15972cdc575cb75f6d57a652a5df623a6a747f26988007579f9d84.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/4e52a3bf8b15972cdc575cb75f6d57a652a5df623a6a747f26988007579f9d84.jpg)

![50dff74d492249219cfbbd95714e2e8c603d91c45b62ae82b3e5e2866f382bf2.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/50dff74d492249219cfbbd95714e2e8c603d91c45b62ae82b3e5e2866f382bf2.jpg)

![52bb7daf130d698be708ce7c2ac32f3a9da9be429a55a2f94deae3d099532d2b.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/52bb7daf130d698be708ce7c2ac32f3a9da9be429a55a2f94deae3d099532d2b.jpg)

![5c6adf94ab42bb3efa0cc74077a72282b65fbb38bbe62c5ca7ee6605cc96f5eb.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/5c6adf94ab42bb3efa0cc74077a72282b65fbb38bbe62c5ca7ee6605cc96f5eb.jpg)

![637d4e8264e124e18f253d1cd8d8118c421180d6c52d20271cf6154cdc1bd0aa.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/637d4e8264e124e18f253d1cd8d8118c421180d6c52d20271cf6154cdc1bd0aa.jpg)

![76d9c273f7cf5338d1038bc2fac56e6b68d736eaf486e65ece92bf6e2a45f657.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/76d9c273f7cf5338d1038bc2fac56e6b68d736eaf486e65ece92bf6e2a45f657.jpg)

![7bfa4525cab998345bd7957f6e9501a0a9614f2006097d5c6f078e339f4647e2.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/7bfa4525cab998345bd7957f6e9501a0a9614f2006097d5c6f078e339f4647e2.jpg)

![86c1b1f47c8634c3a83184925e5547a7d4ac0e8d3e64388074c2e0aeeda4e08e.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/86c1b1f47c8634c3a83184925e5547a7d4ac0e8d3e64388074c2e0aeeda4e08e.jpg)

![a50b93cf917c853d898b41e2c38a6ccdf8e541c83ab2e1a4ceb16902421fb610.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/a50b93cf917c853d898b41e2c38a6ccdf8e541c83ab2e1a4ceb16902421fb610.jpg)

![b346c53d17a5e65091c745b1ba2129d9fd4084cbd8309d5f233503e98b83ec3a.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/images/b346c53d17a5e65091c745b1ba2129d9fd4084cbd8309d5f233503e98b83ec3a.jpg)

### Tables

![0ec4b94b4cee8fabab37a68f9ce1f0a7233532ec250c7f674a2f23cb054fc2bb.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/0ec4b94b4cee8fabab37a68f9ce1f0a7233532ec250c7f674a2f23cb054fc2bb.jpg)

![3075da6e1016a50e6d7af9ab21ed073d2c230d0585cac952ad3472043278446a.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/3075da6e1016a50e6d7af9ab21ed073d2c230d0585cac952ad3472043278446a.jpg)

![3f79053168e6377435d79988b75a743b4e501bae7d8cf1e3a9314c6ed1ae3b83.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/3f79053168e6377435d79988b75a743b4e501bae7d8cf1e3a9314c6ed1ae3b83.jpg)

![53dfd9e5e7c2561caae5155e9ff4d7a5b1b8b4f98dd0ce3f7a6d83f163cc6a56.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/53dfd9e5e7c2561caae5155e9ff4d7a5b1b8b4f98dd0ce3f7a6d83f163cc6a56.jpg)

![59e4350d0465462fad6aeb621141a9a0aefa7cfa1fd556cfc91e846c4e1151e3.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/59e4350d0465462fad6aeb621141a9a0aefa7cfa1fd556cfc91e846c4e1151e3.jpg)

![5d9604f21a5a0beedea5c648d77daeddf88e0d590b7d774d35c0e1e90a37f1d0.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/5d9604f21a5a0beedea5c648d77daeddf88e0d590b7d774d35c0e1e90a37f1d0.jpg)

![81996108f54251dc8a8836182719b101b7e26c3bd6a7f9f855933f750d1ff488.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/81996108f54251dc8a8836182719b101b7e26c3bd6a7f9f855933f750d1ff488.jpg)

![a68949aa17f96e8d7c761d103fb77e1511a3522158bf2c130095e139ffda13f6.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/a68949aa17f96e8d7c761d103fb77e1511a3522158bf2c130095e139ffda13f6.jpg)

![d3557d4e7626ab89cb30a9ad25d201f185c25c54a587f509c6c87f3481c16ee3.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/d3557d4e7626ab89cb30a9ad25d201f185c25c54a587f509c6c87f3481c16ee3.jpg)

![e01d0da0e1dce1d147b4598dcad3d19feff9182343061aaac7d15b4cd7865802.jpg](../icml_results/6309_Affine-Equivariant%20Kernel%20Space%20Encoding%20for%20NeRF%20Editing/tables/e01d0da0e1dce1d147b4598dcad3d19feff9182343061aaac7d15b4cd7865802.jpg)

<a id="paper-6258"></a>

## Can local learning match self-supervised backpropagation?


### Images

![0b7d101fbc7fe6e007ed986a7b2db06d18cea188877845fa147a269c3ded70da.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/0b7d101fbc7fe6e007ed986a7b2db06d18cea188877845fa147a269c3ded70da.jpg)

![845204d3d958bd1dd5628837a747a47d9e9c5fcef97eab39d8ff91fcb55f9247.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/845204d3d958bd1dd5628837a747a47d9e9c5fcef97eab39d8ff91fcb55f9247.jpg)

![94cb505c14a547ca8702c10b52099afbdb55f482300744d8e5208038476ff9fb.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/94cb505c14a547ca8702c10b52099afbdb55f482300744d8e5208038476ff9fb.jpg)

![960c689ff98e19850d4d4d0b9d4e1bbfe6e230ce40589031fed332ff33e67558.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/960c689ff98e19850d4d4d0b9d4e1bbfe6e230ce40589031fed332ff33e67558.jpg)

![d0256e5abb6fa2a1ed70016ae6c13dddb721a78b3d818532ba9e956c6980d86e.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/d0256e5abb6fa2a1ed70016ae6c13dddb721a78b3d818532ba9e956c6980d86e.jpg)

![ded878ce255728f0207389c0fbcbb9e88fae481f972e85aad69a72c9082d2911.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/ded878ce255728f0207389c0fbcbb9e88fae481f972e85aad69a72c9082d2911.jpg)

![e71ce74c53a6bf0c1e3b7a02e86c1232c59973458dcb82ec43ab1d9c52119ef3.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/images/e71ce74c53a6bf0c1e3b7a02e86c1232c59973458dcb82ec43ab1d9c52119ef3.jpg)

### Tables

![1ccc5273f6129fc70d632f1ba3e6110652bc338449ba6307c582daea5c26cc7b.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/1ccc5273f6129fc70d632f1ba3e6110652bc338449ba6307c582daea5c26cc7b.jpg)

![2429bb675b6ca5031aa477bc08312976c02ef525d25304bc9cbe24aae1c4d290.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/2429bb675b6ca5031aa477bc08312976c02ef525d25304bc9cbe24aae1c4d290.jpg)

![712daae4e1eede0a29e9e1cc3d0441bf99e3d646d08af50076014772ab330b5b.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/712daae4e1eede0a29e9e1cc3d0441bf99e3d646d08af50076014772ab330b5b.jpg)

![7a5262312ba123e2254b03afb9488e62ed2bb132e05a174b2a06db398d545e51.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/7a5262312ba123e2254b03afb9488e62ed2bb132e05a174b2a06db398d545e51.jpg)

![8cf5e0eb483dc320f46cfc9cff6a6da8dcb1838f7479afffe3544d9e7d6db6f3.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/8cf5e0eb483dc320f46cfc9cff6a6da8dcb1838f7479afffe3544d9e7d6db6f3.jpg)

![d5b410f0ceef9e4026c9d80c96f23a97d927cadbe2786f42674fd44482c14fd9.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/d5b410f0ceef9e4026c9d80c96f23a97d927cadbe2786f42674fd44482c14fd9.jpg)

![d629ae90a5b119768604d19db3b5303dad6ec80472ad31654d5182f4b3a04edb.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/d629ae90a5b119768604d19db3b5303dad6ec80472ad31654d5182f4b3a04edb.jpg)

![de73bb2a52917e2c5d4b17fb3bdae90c5eb73563b4819f272dd98cfea1b0a9c1.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/de73bb2a52917e2c5d4b17fb3bdae90c5eb73563b4819f272dd98cfea1b0a9c1.jpg)

![e831b9ee501faa1ca2d3f427feae5228abfcc22304edafc49b4ce451fa2a2a21.jpg](../icml_results/6310_Can%20local%20learning%20match%20self-supervised%20backpropagation_/tables/e831b9ee501faa1ca2d3f427feae5228abfcc22304edafc49b4ce451fa2a2a21.jpg)

<a id="paper-6259"></a>

## Near-Optimal Dynamic Matching via Coarsening with Application to Heart Transplantation


### Images

![1aaa8dbd0983d4c5b97d51a930727e1c95946e2a4bdfe79642d3f3364dd19669.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/1aaa8dbd0983d4c5b97d51a930727e1c95946e2a4bdfe79642d3f3364dd19669.jpg)

![1bc2cb29ad33dd50bc4e541808c0786b1f5882f23a305e3b9b0d36d1b31c18f7.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/1bc2cb29ad33dd50bc4e541808c0786b1f5882f23a305e3b9b0d36d1b31c18f7.jpg)

![21b67528a61f88942ec5ddc527ac00c7a9152d13dd9d8a97f88bb65a4b77b3e4.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/21b67528a61f88942ec5ddc527ac00c7a9152d13dd9d8a97f88bb65a4b77b3e4.jpg)

![94d0d97018ee33d42555aa7d960c7446a9bfd24ff86a09fe9a5e3973ffe6d35e.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/94d0d97018ee33d42555aa7d960c7446a9bfd24ff86a09fe9a5e3973ffe6d35e.jpg)

![aeb490c9370266ea429a34f1731ffd47fb62ff9d80c8fc501da7a6f53be7be33.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/aeb490c9370266ea429a34f1731ffd47fb62ff9d80c8fc501da7a6f53be7be33.jpg)

![af10408cd774392996870126241a9dcb3d7efd27722a8ff99fdbc0628b16e0c1.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/af10408cd774392996870126241a9dcb3d7efd27722a8ff99fdbc0628b16e0c1.jpg)

![c7883a1fb5e0aa0e5b35340aea669019bf7afcab9f9216621c5abb4ba1885821.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/c7883a1fb5e0aa0e5b35340aea669019bf7afcab9f9216621c5abb4ba1885821.jpg)

![ed73079b10203d424bac5b96953f7281fa894499fbf229464c376d1c3e5ccf14.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/ed73079b10203d424bac5b96953f7281fa894499fbf229464c376d1c3e5ccf14.jpg)

![f9e5a5dd57ef457b271d3b3ddb47c172aab56fec26e58640f546b41e94e6286c.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/images/f9e5a5dd57ef457b271d3b3ddb47c172aab56fec26e58640f546b41e94e6286c.jpg)

### Tables

![5e0b23dc14b9d3ef7591ec427d3532a617d9f134efc9e70b4f1225365db928af.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/tables/5e0b23dc14b9d3ef7591ec427d3532a617d9f134efc9e70b4f1225365db928af.jpg)

![7d94b54b42ecc755ce89416339ffba1d85f844d66bd0dfe73cc04d7a4b2caeb0.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/tables/7d94b54b42ecc755ce89416339ffba1d85f844d66bd0dfe73cc04d7a4b2caeb0.jpg)

![c8ee1d1006195e0f678ecbd0dd4dc60d55fc2eb6df4207b7a2f44a4f15935e3a.jpg](../icml_results/6311_Near-Optimal%20Dynamic%20Matching%20via%20Coarsening%20with%20Application%20to%20Heart%20Transplantation/tables/c8ee1d1006195e0f678ecbd0dd4dc60d55fc2eb6df4207b7a2f44a4f15935e3a.jpg)

<a id="paper-6260"></a>

## Learning Normalized Energy Models for Linear Inverse Problems


### Images

![0102151c056fb41c325d5220745839c0d2a02042dfa7b62513fec59c63885113.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/0102151c056fb41c325d5220745839c0d2a02042dfa7b62513fec59c63885113.jpg)

![070f3f23db5205c824990f29af39c642a31a3eb797045ba659c5b0a7de9e1f40.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/070f3f23db5205c824990f29af39c642a31a3eb797045ba659c5b0a7de9e1f40.jpg)

![0889a5693305e4eddeef395516bb5607f0c86ff8b0a3e8477f3aebf234935ac0.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/0889a5693305e4eddeef395516bb5607f0c86ff8b0a3e8477f3aebf234935ac0.jpg)

![27f8a4848b7288768015fb9f564f26f295a91abc56dc0cc441d3f9a18f39f3cd.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/27f8a4848b7288768015fb9f564f26f295a91abc56dc0cc441d3f9a18f39f3cd.jpg)

![2b7f82d09977c470cb49499355f0f04f31a6b13abd113b5e2491bf9ea0b637ab.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/2b7f82d09977c470cb49499355f0f04f31a6b13abd113b5e2491bf9ea0b637ab.jpg)

![335801e1da0ca744fa27f2190a92fdf696c8780ffda5cb00d4ed33844c29b55a.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/335801e1da0ca744fa27f2190a92fdf696c8780ffda5cb00d4ed33844c29b55a.jpg)

![4a841ff72c8992e4ddf3e59fc40ddb89e907846923356eaccbeb6bf7899e4eb7.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/4a841ff72c8992e4ddf3e59fc40ddb89e907846923356eaccbeb6bf7899e4eb7.jpg)

![59ced7a5b6e09327f44556ee5d54f1505160f6272eaa074189605d0e94f38626.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/59ced7a5b6e09327f44556ee5d54f1505160f6272eaa074189605d0e94f38626.jpg)

![6864230ae3e646057b7cf9ac437e1cf33847f752f4b1880f02a67e31b0b35e1c.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/6864230ae3e646057b7cf9ac437e1cf33847f752f4b1880f02a67e31b0b35e1c.jpg)

![6969c51fb90e8a0c06f203aa29121824866825015a64af4814e9b358eaaccebc.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/6969c51fb90e8a0c06f203aa29121824866825015a64af4814e9b358eaaccebc.jpg)

![6a0a6c09f6e043d3dfcf695dd1fc7a303811b97241c4483bbf1f5a09c690ec04.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/6a0a6c09f6e043d3dfcf695dd1fc7a303811b97241c4483bbf1f5a09c690ec04.jpg)

![7ca7e64f8189d2675ed46e89c86abadaf996ccdae5ecf9921ec5c41b7cf8433a.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/7ca7e64f8189d2675ed46e89c86abadaf996ccdae5ecf9921ec5c41b7cf8433a.jpg)

![8684bb59fff296ca62fe98041ce63f659c258e2028a414efff87469ebadd9974.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/8684bb59fff296ca62fe98041ce63f659c258e2028a414efff87469ebadd9974.jpg)

![88ba18f809d94e969a71d0e9d21b46963fc522f4afcff9b8f7b49ee5c469619f.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/88ba18f809d94e969a71d0e9d21b46963fc522f4afcff9b8f7b49ee5c469619f.jpg)

![8c9da9900b58fbf4d4cec969d003d40d1c3fe587dba9bab6e0a079b7db4e0884.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/8c9da9900b58fbf4d4cec969d003d40d1c3fe587dba9bab6e0a079b7db4e0884.jpg)

![9c508058653c04cac0fcbbe45e057fdaa372e43e3a8be89945439b6c91de6408.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/9c508058653c04cac0fcbbe45e057fdaa372e43e3a8be89945439b6c91de6408.jpg)

![9e94931b3cc5e9d30b33d8bdd4bee969e6d3116ed26a35711a3b04b168842bf9.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/9e94931b3cc5e9d30b33d8bdd4bee969e6d3116ed26a35711a3b04b168842bf9.jpg)

![a63a411f4d10893abf761e9a7b9d80e3f956263a8cf20d1c36969933e76f3957.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/a63a411f4d10893abf761e9a7b9d80e3f956263a8cf20d1c36969933e76f3957.jpg)

![a6f387f4d8eb845a8239e3f09e31b452695262557863ed296f89060ed66b8647.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/a6f387f4d8eb845a8239e3f09e31b452695262557863ed296f89060ed66b8647.jpg)

![b603a5c8b801b0e0a6c88388da2898547d30cc592afe31a2f10a0ce0c7c9ee54.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/b603a5c8b801b0e0a6c88388da2898547d30cc592afe31a2f10a0ce0c7c9ee54.jpg)

![ba8bcdd5dc2f5dbe30350789c59dff1f82a89abeeaef2a38f4565a8d483688a0.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/ba8bcdd5dc2f5dbe30350789c59dff1f82a89abeeaef2a38f4565a8d483688a0.jpg)

![ba9f14389655d6b45175d923178dd7621e4c32da3077715953a1a0bfd85c89c1.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/ba9f14389655d6b45175d923178dd7621e4c32da3077715953a1a0bfd85c89c1.jpg)

![baf4f1144879e20f58c134d302d30d2c4df1b94bc2ba27e997c7a67b07e0937d.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/baf4f1144879e20f58c134d302d30d2c4df1b94bc2ba27e997c7a67b07e0937d.jpg)

![c52c617b8366567bcf6ecd12ce1d473df348a1dd10f1edf161d29848553df08a.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/c52c617b8366567bcf6ecd12ce1d473df348a1dd10f1edf161d29848553df08a.jpg)

![c6ae3a12257408691e5274a06f763fdc2fdbca788cf512413714b196c4195ecb.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/c6ae3a12257408691e5274a06f763fdc2fdbca788cf512413714b196c4195ecb.jpg)

![d662a617057e0112f54bd2d22c6ea17c4bf20de2a4e6e8a3dd45b1072a652c48.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/d662a617057e0112f54bd2d22c6ea17c4bf20de2a4e6e8a3dd45b1072a652c48.jpg)

![d754ebd48e69b1f908753ec7e5533e9bf52937b529b08079288bed17d387310d.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/d754ebd48e69b1f908753ec7e5533e9bf52937b529b08079288bed17d387310d.jpg)

![da9ba716b8917589c915e67f2698e809bf772cb867f81b8451ebabea99bdcf10.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/da9ba716b8917589c915e67f2698e809bf772cb867f81b8451ebabea99bdcf10.jpg)

![ec6ed5d58747f1b7da3564a0888e629a2c6e8c5488d391a2875abcc416b6660b.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/images/ec6ed5d58747f1b7da3564a0888e629a2c6e8c5488d391a2875abcc416b6660b.jpg)

### Tables

![1fd45a49559a6a97f50a38b31f2cec6d0ea301ccbbe716f6953bbb31ce1000c6.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/1fd45a49559a6a97f50a38b31f2cec6d0ea301ccbbe716f6953bbb31ce1000c6.jpg)

![319a78ad60ee85d579651e8a09be4cd894c85a43d28a573ea8160d2a297d9d5e.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/319a78ad60ee85d579651e8a09be4cd894c85a43d28a573ea8160d2a297d9d5e.jpg)

![452f5c34604ce665b29fc710478deb538fe3cb5c46ead7135377a5a2284273e6.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/452f5c34604ce665b29fc710478deb538fe3cb5c46ead7135377a5a2284273e6.jpg)

![511ebfe5fa4e4c97c5879650c4e30e36f7f1e9b5b977449b38b948e980c27131.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/511ebfe5fa4e4c97c5879650c4e30e36f7f1e9b5b977449b38b948e980c27131.jpg)

![6e10b64255d02b104653bfafc02001eb64ead1ea782d7667a0bd135f2c470155.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/6e10b64255d02b104653bfafc02001eb64ead1ea782d7667a0bd135f2c470155.jpg)

![6f6509689f9da1f7c4e254bad23e99d06be201f88a71597de3ef62dd9f53e64d.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/6f6509689f9da1f7c4e254bad23e99d06be201f88a71597de3ef62dd9f53e64d.jpg)

![87730e31e4aaba2f003dc834424e0cce280e033c288fd8d37494d5a2f871fe28.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/87730e31e4aaba2f003dc834424e0cce280e033c288fd8d37494d5a2f871fe28.jpg)

![92a679a1b181f6a2454cc4bbd5aef559b0dc4f2a2089a42290a3f48b747f760e.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/92a679a1b181f6a2454cc4bbd5aef559b0dc4f2a2089a42290a3f48b747f760e.jpg)

![967a314fb94eaa21cdb80d451297f872ea2b75a587cd2ff381c410e6308f8c36.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/967a314fb94eaa21cdb80d451297f872ea2b75a587cd2ff381c410e6308f8c36.jpg)

![9c95608317b43c26d621774881ae326fcd754081b0a9802c3369244fcad69553.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/9c95608317b43c26d621774881ae326fcd754081b0a9802c3369244fcad69553.jpg)

![a26e0e9a610a45f47a2375887ee9d1f582cea4a489f7aa8090ae8e8020cb6e6d.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/a26e0e9a610a45f47a2375887ee9d1f582cea4a489f7aa8090ae8e8020cb6e6d.jpg)

![cd8859e4a09f11f553a2ec9f96931fdf6479e6a026688ae6dc5d4dcad1b9e4ea.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/cd8859e4a09f11f553a2ec9f96931fdf6479e6a026688ae6dc5d4dcad1b9e4ea.jpg)

![db5967c762bad0258c562f735e8eaf45b96ba6d2e7e2d75f4751ba1a46279af0.jpg](../icml_results/6312_Learning%20Normalized%20Energy%20Models%20for%20Linear%20Inverse%20Problems/tables/db5967c762bad0258c562f735e8eaf45b96ba6d2e7e2d75f4751ba1a46279af0.jpg)

<a id="paper-6261"></a>

## PAC-Bayesian Reinforcement Learning Trains Generalizable Policies


### Images

![2faaf4529e12da72b1069ca74df284333c38a244415448f6ea1b64486672301e.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/2faaf4529e12da72b1069ca74df284333c38a244415448f6ea1b64486672301e.jpg)

![333531fb7d1cb70c2363410e151e56c32a93a95ea2a23319516e4d4db29b33d0.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/333531fb7d1cb70c2363410e151e56c32a93a95ea2a23319516e4d4db29b33d0.jpg)

![4115afb4f22da3f5123db4d4fa8ca1223c51b8d4aebf783cac79420993b81dcf.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/4115afb4f22da3f5123db4d4fa8ca1223c51b8d4aebf783cac79420993b81dcf.jpg)

![729db7b85c98f142eb12c07b6ef05233005de05ee926188180d5b771a7a8793e.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/729db7b85c98f142eb12c07b6ef05233005de05ee926188180d5b771a7a8793e.jpg)

![9836a5f0c67e009ec7c251d1526cae5338ebb301e16cf7989786c20f271ab85b.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/9836a5f0c67e009ec7c251d1526cae5338ebb301e16cf7989786c20f271ab85b.jpg)

![a94e4532411949f61438c763563a8bc9015418b5d39b1b6ec89e732eb4133a07.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/a94e4532411949f61438c763563a8bc9015418b5d39b1b6ec89e732eb4133a07.jpg)

![ab93a8c9cd587c989d73b72e3c090fa06b35dc8edfa84e774821a57c6ba7c141.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/ab93a8c9cd587c989d73b72e3c090fa06b35dc8edfa84e774821a57c6ba7c141.jpg)

![afbcbea0e944c5cc08fe42514974313b5b24a7f9ae4f962ad0e08e401ef6435d.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/afbcbea0e944c5cc08fe42514974313b5b24a7f9ae4f962ad0e08e401ef6435d.jpg)

![d28fd1cfdd4586a32e67ea06faff47fde48819565eed466231757e29a8a19927.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/images/d28fd1cfdd4586a32e67ea06faff47fde48819565eed466231757e29a8a19927.jpg)

### Tables

![71bc6f6e50e066c14ba9bf24065ac3653c26acf117f4fc2c114128aa842f9eaa.jpg](../icml_results/6313_PAC-Bayesian%20Reinforcement%20Learning%20Trains%20Generalizable%20Policies/tables/71bc6f6e50e066c14ba9bf24065ac3653c26acf117f4fc2c114128aa842f9eaa.jpg)

<a id="paper-6262"></a>

## One-Step Gradient Delay is Not a Barrier for Large-Scale Asynchronous Pipeline Parallel LLM Pretraining


### Images

![07ed0d76830d4aefada1796a01223c87f6dbbda99e8ac129e32ad3c6a849804e.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/07ed0d76830d4aefada1796a01223c87f6dbbda99e8ac129e32ad3c6a849804e.jpg)

![0e736dc26a810f8811f516bac7acdd8cc2c73a5dc3e12fb313f02b5bf2c7ba2b.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/0e736dc26a810f8811f516bac7acdd8cc2c73a5dc3e12fb313f02b5bf2c7ba2b.jpg)

![2ba27edafa9c7b653228c73021a1aaf863d313692935d2cd18abb822722e9505.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/2ba27edafa9c7b653228c73021a1aaf863d313692935d2cd18abb822722e9505.jpg)

![4a16056d10827b5b36a8628258b10da00d15c801e27f3c09f89aa332f23c8425.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/4a16056d10827b5b36a8628258b10da00d15c801e27f3c09f89aa332f23c8425.jpg)

![5e706cf53d84264d0a7bfc14cb93e5835c29cd95d2281c05e473dcfd60159460.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/5e706cf53d84264d0a7bfc14cb93e5835c29cd95d2281c05e473dcfd60159460.jpg)

![7c4ff7306409c0f57e6f7ea38a324b11f071b0a5245f5ede2e8bd778ffc56148.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/7c4ff7306409c0f57e6f7ea38a324b11f071b0a5245f5ede2e8bd778ffc56148.jpg)

![90dc41f98ee86f176a214711b441a5d4a4edaff2322159d4b4773f162c613340.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/90dc41f98ee86f176a214711b441a5d4a4edaff2322159d4b4773f162c613340.jpg)

![9c2e7e93ff7bed46a4a43ebe686b4fab5ac042a4d8af25c6533898c2b628028a.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/9c2e7e93ff7bed46a4a43ebe686b4fab5ac042a4d8af25c6533898c2b628028a.jpg)

![abb80108f7a6da012e100afc011abf60291dab40cd419eabb09c7c9cbba09fa9.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/abb80108f7a6da012e100afc011abf60291dab40cd419eabb09c7c9cbba09fa9.jpg)

![cf8a42bea4f556a9a82180215c2608e3d6c74a96a3e76b269891d1dcf72e3e62.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/cf8a42bea4f556a9a82180215c2608e3d6c74a96a3e76b269891d1dcf72e3e62.jpg)

![d1a9f995bbf9d135bc2335ece5d9e76cd569c548f7eba0ca82c8eefe4feac73f.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/d1a9f995bbf9d135bc2335ece5d9e76cd569c548f7eba0ca82c8eefe4feac73f.jpg)

![e0cb673e02d785b921b73fb44019b2094bcd7eb439197c006cfff0c8a3974c56.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/e0cb673e02d785b921b73fb44019b2094bcd7eb439197c006cfff0c8a3974c56.jpg)

![fb14822083ec4d85a73d1e85c2e5fc9c73ac8dd78a0f8ec83a5d85cd5a4995bd.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/images/fb14822083ec4d85a73d1e85c2e5fc9c73ac8dd78a0f8ec83a5d85cd5a4995bd.jpg)

### Tables

![0ff3fcf34b6fd383524ef6e47c0e5f1a05f48db6a753ac055f6e73045082d326.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/0ff3fcf34b6fd383524ef6e47c0e5f1a05f48db6a753ac055f6e73045082d326.jpg)

![109e97d2c737b22139c6879d158d647e57e97aec03c1812f0548f38d333e30fc.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/109e97d2c737b22139c6879d158d647e57e97aec03c1812f0548f38d333e30fc.jpg)

![234d49fb951e02dae3c239afaeada20a0d6e2e1b786b3f8eebb93d1aecaf5809.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/234d49fb951e02dae3c239afaeada20a0d6e2e1b786b3f8eebb93d1aecaf5809.jpg)

![2caa6332a0e6d346fd3e50b0d0d1e72ba9b97ccec8a262d4b98ae2ea75ea2059.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/2caa6332a0e6d346fd3e50b0d0d1e72ba9b97ccec8a262d4b98ae2ea75ea2059.jpg)

![4657965376055496292f254d66e49aadb7974f77caec631251306660db7c8036.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/4657965376055496292f254d66e49aadb7974f77caec631251306660db7c8036.jpg)

![6b48c903d7fa6fd886c14ada77d482c25a488d66225f79c5489f343ae7f35d05.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/6b48c903d7fa6fd886c14ada77d482c25a488d66225f79c5489f343ae7f35d05.jpg)

![bf738d292748c5e721cb15e5c9503dc099d1f9c98a7f37c91c691c9d269c59e4.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/bf738d292748c5e721cb15e5c9503dc099d1f9c98a7f37c91c691c9d269c59e4.jpg)

![c58705e9f623f07b5cc0bc1dc647af022a61846d8665e518834333fc3d18d3e9.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/c58705e9f623f07b5cc0bc1dc647af022a61846d8665e518834333fc3d18d3e9.jpg)

![cf5f810d5a6c6b977291a4dae74163a00ad44712de03e9b8de27983d018450b8.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/cf5f810d5a6c6b977291a4dae74163a00ad44712de03e9b8de27983d018450b8.jpg)

![d029abbb960559de470beb1c4334606b560bd84a70c1f20e2f7b1e9568609533.jpg](../icml_results/6314_One-Step%20Gradient%20Delay%20is%20Not%20a%20Barrier%20for%20Large-Scale%20Asynchronous%20Pipeline%20Parallel%20LLM%20Pretrain/tables/d029abbb960559de470beb1c4334606b560bd84a70c1f20e2f7b1e9568609533.jpg)

<a id="paper-6263"></a>

## HaloProbe: Bayesian Detection and Mitigation of Object Hallucinations in Vision-Language Models


### Images

![0989e01cccb30fc90c56bab31aac4b80c7efe8043b642fb4b3068d72715fdb8f.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/0989e01cccb30fc90c56bab31aac4b80c7efe8043b642fb4b3068d72715fdb8f.jpg)

![1c9801ec26eabb33137ff68f46816712a1d4e384f0b24ceb668ad1d3ea94bb94.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/1c9801ec26eabb33137ff68f46816712a1d4e384f0b24ceb668ad1d3ea94bb94.jpg)

![4fa3c107c012ff20149b8712c49cde2bd87a0026ce0fe35a296d03ec98e5e373.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/4fa3c107c012ff20149b8712c49cde2bd87a0026ce0fe35a296d03ec98e5e373.jpg)

![5590ced4dc6bbc0ff1270d9017f2eb9525f0b57cae59b9d913967ccdb25a4a93.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/5590ced4dc6bbc0ff1270d9017f2eb9525f0b57cae59b9d913967ccdb25a4a93.jpg)

![5f86f6da14d2f3acdbb1cf20b40755e5fce24c4d58412089590067a068f507bf.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/5f86f6da14d2f3acdbb1cf20b40755e5fce24c4d58412089590067a068f507bf.jpg)

![6812a88945524d02d31e0fff0a2bdaa42bbd4ee42da425fa3dc432c8c0c5c5c6.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/6812a88945524d02d31e0fff0a2bdaa42bbd4ee42da425fa3dc432c8c0c5c5c6.jpg)

![7180183d4184f3315aa3a0cbfe2dac3cdf11eaf93cdbdedb52e445eefb6a9af7.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/7180183d4184f3315aa3a0cbfe2dac3cdf11eaf93cdbdedb52e445eefb6a9af7.jpg)

![84e65c568df6e77699e0d7e62d73f7d87599f3230d3330f17f13cfcd4f36e687.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/84e65c568df6e77699e0d7e62d73f7d87599f3230d3330f17f13cfcd4f36e687.jpg)

![905dba387daa6c0e00a04e533d6f208666680e954b0a6f23808072e3db9f4f76.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/905dba387daa6c0e00a04e533d6f208666680e954b0a6f23808072e3db9f4f76.jpg)

![a73e35d2d788c622d0d54d6164168985ba6902248a27ce0460c0d7fa4e4958f9.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/a73e35d2d788c622d0d54d6164168985ba6902248a27ce0460c0d7fa4e4958f9.jpg)

![a79e65aeec1e037eedbd962987074fedf5878d5d2b529df378a190cd391f97fa.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/a79e65aeec1e037eedbd962987074fedf5878d5d2b529df378a190cd391f97fa.jpg)

![a890b9999a203fe7514ac90c8d0e2fba2475352684c474b669edd95cf2aa340c.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/a890b9999a203fe7514ac90c8d0e2fba2475352684c474b669edd95cf2aa340c.jpg)

![dff245860d2e1e380ed9ef764ab7d05249a52a32953dd140c34f7d89e94e9890.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/dff245860d2e1e380ed9ef764ab7d05249a52a32953dd140c34f7d89e94e9890.jpg)

![e96e4b21c0c951c4a7552e67043338e753b90ac2571cc8380838549dc16c359a.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/e96e4b21c0c951c4a7552e67043338e753b90ac2571cc8380838549dc16c359a.jpg)

![e9e932b671347583b2053f3971988ac21e429c651511cadf50d866bebcda061f.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/e9e932b671347583b2053f3971988ac21e429c651511cadf50d866bebcda061f.jpg)

![f1f0ff353592864dbe559665a0465b24d78f615b09413647d19b69108a01f90d.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/images/f1f0ff353592864dbe559665a0465b24d78f615b09413647d19b69108a01f90d.jpg)

### Tables

![20971ebd131382de8a6f8cc4bf746951670c7971da2cb038a6d8dba04f7c7f06.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/20971ebd131382de8a6f8cc4bf746951670c7971da2cb038a6d8dba04f7c7f06.jpg)

![2d0a5df2218ba93200487811dad110d29a5d7e09f494d8b3943fbb34a60c0add.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/2d0a5df2218ba93200487811dad110d29a5d7e09f494d8b3943fbb34a60c0add.jpg)

![4f458f9826bb712463489bed832871771871a28bf119d3eed9594ec47771424a.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/4f458f9826bb712463489bed832871771871a28bf119d3eed9594ec47771424a.jpg)

![59a3e02e67a99f6f9ebecb9293d03f259355afdbb0002c990f4e87000db2c5eb.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/59a3e02e67a99f6f9ebecb9293d03f259355afdbb0002c990f4e87000db2c5eb.jpg)

![5f38200541c6cd0750326e966eadeb4bbbcf6e39cfc9e4fafb50496e490269b1.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/5f38200541c6cd0750326e966eadeb4bbbcf6e39cfc9e4fafb50496e490269b1.jpg)

![6aaa8719d98df7d37feb9d7b5b393b42558a00f0e6d8a2abe3c9f6ad789b5158.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/6aaa8719d98df7d37feb9d7b5b393b42558a00f0e6d8a2abe3c9f6ad789b5158.jpg)

![7311b9be5ac6b05b58d55eb3b21ddbf0b54948e2dc6b0166f694b8ad63fbba7f.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/7311b9be5ac6b05b58d55eb3b21ddbf0b54948e2dc6b0166f694b8ad63fbba7f.jpg)

![89782155d1945f2e2f5c77575548aec9ef87a30c6caa3e76289c2bfa7b03b774.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/89782155d1945f2e2f5c77575548aec9ef87a30c6caa3e76289c2bfa7b03b774.jpg)

![a49ed0bbd5fb5d6492b8978d487a81c0e501672a5e0dad9ae70e1eb9a7f81088.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/a49ed0bbd5fb5d6492b8978d487a81c0e501672a5e0dad9ae70e1eb9a7f81088.jpg)

![a6ea1a3647e00d3adb24d7a5bb970a22260f718e19e6fd0517d1a476f3963bea.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/a6ea1a3647e00d3adb24d7a5bb970a22260f718e19e6fd0517d1a476f3963bea.jpg)

![e60b0351377c539260fec3051a3de8753a36b8a1ce53a32ba5d4038eea41d151.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/e60b0351377c539260fec3051a3de8753a36b8a1ce53a32ba5d4038eea41d151.jpg)

![ec0046b1db4c1e69b9c1a5c55f4763a8d26d0683356dbab531569b960d17fe97.jpg](../icml_results/6315_HaloProbe_%20Bayesian%20Detection%20and%20Mitigation%20of%20Object%20Hallucinations%20in%20Vision-Language%20Models/tables/ec0046b1db4c1e69b9c1a5c55f4763a8d26d0683356dbab531569b960d17fe97.jpg)

<a id="paper-6264"></a>

## Conflict-Aware Adaptive Alignment for LLM Hallucination Mitigation


### Images

![38f1f1c18dbfa6e2a18ef79e8b99fefd98fb41b0d0b79e9f8f6ff75d1e3392a0.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/images/38f1f1c18dbfa6e2a18ef79e8b99fefd98fb41b0d0b79e9f8f6ff75d1e3392a0.jpg)

![48017002ae8156c6ab6a8e7bec6f89889432550c5d07f7fa6e00b8d516437103.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/images/48017002ae8156c6ab6a8e7bec6f89889432550c5d07f7fa6e00b8d516437103.jpg)

![5875a535fe7dcc614cad3a51ee01431bbe13d944898d0512422db1812e7b4cf8.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/images/5875a535fe7dcc614cad3a51ee01431bbe13d944898d0512422db1812e7b4cf8.jpg)

![b9efcc24019166f344238943630ad6a5e7bad8e37540786ff803f7759931f9e5.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/images/b9efcc24019166f344238943630ad6a5e7bad8e37540786ff803f7759931f9e5.jpg)

![c0dcf6309412fe4eca11eaf0038e0743d07d6f8690e573d4e12ca35dc1e63c20.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/images/c0dcf6309412fe4eca11eaf0038e0743d07d6f8690e573d4e12ca35dc1e63c20.jpg)

### Tables

![0087fcfe42f6bac037fcd886dab46bcd919b9671852fcc20193df091b5a51779.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/0087fcfe42f6bac037fcd886dab46bcd919b9671852fcc20193df091b5a51779.jpg)

![1212be187541e0aa1253e875c849c3ee81a307839022f8caa59820a34ba76695.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/1212be187541e0aa1253e875c849c3ee81a307839022f8caa59820a34ba76695.jpg)

![2232b5891d519e90034891aadd48057ddfa120aab761dc419217b5a1ffdfd628.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/2232b5891d519e90034891aadd48057ddfa120aab761dc419217b5a1ffdfd628.jpg)

![2f6b7dd83d2bac960f75eea0d8f3cc0c5f96645633108d6fab176e172a4e172f.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/2f6b7dd83d2bac960f75eea0d8f3cc0c5f96645633108d6fab176e172a4e172f.jpg)

![4d95d69b9f82d7081d1725888ceccd4992d2be72b28d3f36b7d95ccc2b36de17.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/4d95d69b9f82d7081d1725888ceccd4992d2be72b28d3f36b7d95ccc2b36de17.jpg)

![a429d1dfd7bec7ef9d71ad5dc47a521b0111e97a8aa7129e5695ef9903ca0b4a.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/a429d1dfd7bec7ef9d71ad5dc47a521b0111e97a8aa7129e5695ef9903ca0b4a.jpg)

![a65b0c58499c674c3c0494caa610e93935aa53275c69de826c5dab3e723a150b.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/a65b0c58499c674c3c0494caa610e93935aa53275c69de826c5dab3e723a150b.jpg)

![c4103ce530631ef43ef0edd503c2263ce0c626db0f8e7de2bd81e17167519134.jpg](../icml_results/6316_Conflict-Aware%20Adaptive%20Alignment%20for%20LLM%20Hallucination%20Mitigation/tables/c4103ce530631ef43ef0edd503c2263ce0c626db0f8e7de2bd81e17167519134.jpg)

<a id="paper-6265"></a>

## Less Token, More Signal: MoE Expert Pruning via Critical Token Selection


### Images

![127c88404fa1df73654b002bfebbdcf583cf59114473e3614a9a019c595543be.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/127c88404fa1df73654b002bfebbdcf583cf59114473e3614a9a019c595543be.jpg)

![18739f8f8dc5ac64f86861b6d2ac3eb7ca51cb489f817974c09910235fb85fe9.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/18739f8f8dc5ac64f86861b6d2ac3eb7ca51cb489f817974c09910235fb85fe9.jpg)

![323c949ffa4d0688ec523997a7ce8ae3e16e41b94d8476bd7333b80277b194ee.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/323c949ffa4d0688ec523997a7ce8ae3e16e41b94d8476bd7333b80277b194ee.jpg)

![3b66cdcdee680d342321ffac19c5273cdf3023e5ba414f129be60bf73af9dbd8.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/3b66cdcdee680d342321ffac19c5273cdf3023e5ba414f129be60bf73af9dbd8.jpg)

![4b92f7b7c3e376f8086fb0e70189f6e2bc9432242f2d97de76117c3615305059.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/4b92f7b7c3e376f8086fb0e70189f6e2bc9432242f2d97de76117c3615305059.jpg)

![70cd0d9f82c35dcd53fa62046614b832ee1b188f119f2ae124588058411fb12e.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/70cd0d9f82c35dcd53fa62046614b832ee1b188f119f2ae124588058411fb12e.jpg)

![94ac8c6b5e725c4f8c6af34b151b95f4858ad2feda0ad76c1a022d1160076fcc.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/94ac8c6b5e725c4f8c6af34b151b95f4858ad2feda0ad76c1a022d1160076fcc.jpg)

![96171e81d2381f6715f43336c9da085ef128e994d47c84531a19207fb81a0f52.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/96171e81d2381f6715f43336c9da085ef128e994d47c84531a19207fb81a0f52.jpg)

![a6d3c2b71b6b5d1b89a63de74373bef3d8c8b25da3322b1413f90eba59b2331b.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/a6d3c2b71b6b5d1b89a63de74373bef3d8c8b25da3322b1413f90eba59b2331b.jpg)

![abf96cd2cead37747232a7bc6156666b9a7451e412ec80d0bda0217b70fafebf.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/abf96cd2cead37747232a7bc6156666b9a7451e412ec80d0bda0217b70fafebf.jpg)

![bf94e4bf823cc697167235774b83289e2caa67110d9994ecf6ee098307383dc7.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/bf94e4bf823cc697167235774b83289e2caa67110d9994ecf6ee098307383dc7.jpg)

![cdd3cd36ec4de93ecf692333700826f5c569f87cb402eb3c083af022613d9503.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/cdd3cd36ec4de93ecf692333700826f5c569f87cb402eb3c083af022613d9503.jpg)

![da4d23f85fadfbdbda3fe0a35711939171c524f2d32ced4c6c12eae8db08b863.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/da4d23f85fadfbdbda3fe0a35711939171c524f2d32ced4c6c12eae8db08b863.jpg)

![ed7259cf61fca5279411b6e2ae7fd7aa300ae495cfeecfe0f515b81d9ad31d54.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/images/ed7259cf61fca5279411b6e2ae7fd7aa300ae495cfeecfe0f515b81d9ad31d54.jpg)

### Tables

![0db89f25f083341f088b752d12d5d7d6e9e2bca78f7e2a256191625051c303f7.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/0db89f25f083341f088b752d12d5d7d6e9e2bca78f7e2a256191625051c303f7.jpg)

![49b403d6b2670781fbde62c3b881946cee529ae6cce8367a304ce199cae9b07b.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/49b403d6b2670781fbde62c3b881946cee529ae6cce8367a304ce199cae9b07b.jpg)

![6c279d96e9da30c8f80b00e679e317bed6c6ff7574edf7b3a533abaa5a7e7e83.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/6c279d96e9da30c8f80b00e679e317bed6c6ff7574edf7b3a533abaa5a7e7e83.jpg)

![772073daeba2e639c3ffedfd1d43a8ea6d7dee12de5a08c957493df0b17dea79.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/772073daeba2e639c3ffedfd1d43a8ea6d7dee12de5a08c957493df0b17dea79.jpg)

![7cd59c0e3620737c87d003c9776f7f39895342e59363b612b14f4f41ba72b797.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/7cd59c0e3620737c87d003c9776f7f39895342e59363b612b14f4f41ba72b797.jpg)

![851bd9e1b72069ff130a2ad6ccaaa46acac876152493c09631248c27c40b0aa0.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/851bd9e1b72069ff130a2ad6ccaaa46acac876152493c09631248c27c40b0aa0.jpg)

![867d2104e2d5630b8b0c82cd07cdb83f706a994938b4cbcff3646f30093da137.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/867d2104e2d5630b8b0c82cd07cdb83f706a994938b4cbcff3646f30093da137.jpg)

![ad0fbe4c34ec292bd6ed1f299054d093cd36ce48283ebc69bbdab0becc006645.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/ad0fbe4c34ec292bd6ed1f299054d093cd36ce48283ebc69bbdab0becc006645.jpg)

![cba1e89c2118a6faaa7c5775a43e01eee5c95ca3fbdd9dbc3148d374c1213859.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/cba1e89c2118a6faaa7c5775a43e01eee5c95ca3fbdd9dbc3148d374c1213859.jpg)

![daa0dd2c9acc6b5e719cb869af565f396991658e2883b8d8d2e8ffc613f754dd.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/daa0dd2c9acc6b5e719cb869af565f396991658e2883b8d8d2e8ffc613f754dd.jpg)

![ddb435d8aac77e7661d5dc3f7dedee05b901053fd03db55431659b60fc7b4f74.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/ddb435d8aac77e7661d5dc3f7dedee05b901053fd03db55431659b60fc7b4f74.jpg)

![e276ca0431eb44bbcd19aa3f7753190c32c5505583a9eb848e56c3b35dcbbc04.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/e276ca0431eb44bbcd19aa3f7753190c32c5505583a9eb848e56c3b35dcbbc04.jpg)

![eb817d9ebc7bd4281b0bf7ee12eb23359f21f5e2b9c51eeb27ca6336fa3754c9.jpg](../icml_results/6317_Less%20Token%2C%20More%20Signal_%20MoE%20Expert%20Pruning%20via%20Critical%20Token%20Selection/tables/eb817d9ebc7bd4281b0bf7ee12eb23359f21f5e2b9c51eeb27ca6336fa3754c9.jpg)

<a id="paper-6266"></a>

## On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM agents


### Images

![0c956001193efa7a192388f19877f12bb1bd01ccd3fe44bee83a907527a06055.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/0c956001193efa7a192388f19877f12bb1bd01ccd3fe44bee83a907527a06055.jpg)

![10ca81276d11ffcbbe6e27c776235052b95ce220d10dc71b6f6322a7dfb9c1a2.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/10ca81276d11ffcbbe6e27c776235052b95ce220d10dc71b6f6322a7dfb9c1a2.jpg)

![4ab5caef043f5c3a63fbc84930dbb929606ffdee4f0f211918676778601ae7f7.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/4ab5caef043f5c3a63fbc84930dbb929606ffdee4f0f211918676778601ae7f7.jpg)

![773b8f3bf2bcbf81705bbe68bbb3df07ac75610dd38da4f396faf71f041b7412.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/773b8f3bf2bcbf81705bbe68bbb3df07ac75610dd38da4f396faf71f041b7412.jpg)

![af5bdf7d8375b8d35fbf8c7a9ccd1f2f63f77cbbeba3d1f98a013659d93783e5.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/af5bdf7d8375b8d35fbf8c7a9ccd1f2f63f77cbbeba3d1f98a013659d93783e5.jpg)

![b384969e0ba3813b0cf53f2555423027f735009dc41b799c564d80e53d352a7c.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/b384969e0ba3813b0cf53f2555423027f735009dc41b799c564d80e53d352a7c.jpg)

![cd716f075bebfaad6ec1398538b2e3256c7c89a0b8567df5de93aa0e95465735.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/images/cd716f075bebfaad6ec1398538b2e3256c7c89a0b8567df5de93aa0e95465735.jpg)

### Tables

![9de2db3062ea567a788b86f130c990e5fc693785fed8df2cfbafe886403cf841.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/tables/9de2db3062ea567a788b86f130c990e5fc693785fed8df2cfbafe886403cf841.jpg)

![ef939870d3d28c4ef4fce2689aa08a992b22c148d6904522390d2432886276ac.jpg](../icml_results/6318_On%20Information%20Self-Locking%20in%20Reinforcement%20Learning%20for%20Active%20Reasoning%20of%20LLM%20agents/tables/ef939870d3d28c4ef4fce2689aa08a992b22c148d6904522390d2432886276ac.jpg)

<a id="paper-6267"></a>

## DARC: Disagreement-Aware Alignment via Risk-Constrained Decoding


### Images

![11f59cd145046d215419e2be6f8db770d3063bfa6e7ca64424d8e3e016933dc2.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/images/11f59cd145046d215419e2be6f8db770d3063bfa6e7ca64424d8e3e016933dc2.jpg)

![526eb2979a71345f33d6d95a8f6d0ed02596ba5e85354ab314db75f90d8f60db.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/images/526eb2979a71345f33d6d95a8f6d0ed02596ba5e85354ab314db75f90d8f60db.jpg)

![835abc2cf63df39c6daf9f014469de7222692c1d426ba5043f7b2534f1c20e83.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/images/835abc2cf63df39c6daf9f014469de7222692c1d426ba5043f7b2534f1c20e83.jpg)

![878354af753a79e2dc66615ffbc3b6e689ebff333c784151d3ce0918cadb3937.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/images/878354af753a79e2dc66615ffbc3b6e689ebff333c784151d3ce0918cadb3937.jpg)

![a864c8f40820f4eb27ccb8873a9666acede56647036bdccdd42a9478c8d89a1b.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/images/a864c8f40820f4eb27ccb8873a9666acede56647036bdccdd42a9478c8d89a1b.jpg)

![d74d87dc1e69b144892c054ebd6ce2df4c63a7bc209cacab1b832e1a86533380.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/images/d74d87dc1e69b144892c054ebd6ce2df4c63a7bc209cacab1b832e1a86533380.jpg)

### Tables

![0c756dded5ffaed2cbe35683e25cca0ad4ae61f81463894cce3c5d23af3175fc.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/0c756dded5ffaed2cbe35683e25cca0ad4ae61f81463894cce3c5d23af3175fc.jpg)

![22895ec797a4efb23ce306cd5ff9cab40df150cc99e91bccfd8bf94300ba3246.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/22895ec797a4efb23ce306cd5ff9cab40df150cc99e91bccfd8bf94300ba3246.jpg)

![2cf3e9b8a5e900d9366333d21e07f7c8ae24079e284b71a3f3dfef15978d0c07.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/2cf3e9b8a5e900d9366333d21e07f7c8ae24079e284b71a3f3dfef15978d0c07.jpg)

![38a4021f54ad2dec48a15777660565ab0f474df563685b0cedd92d778d7763ae.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/38a4021f54ad2dec48a15777660565ab0f474df563685b0cedd92d778d7763ae.jpg)

![45cea4b1cc60d43e5dfaa3ffb694cf92737cca167cbf05b47880757846a8400e.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/45cea4b1cc60d43e5dfaa3ffb694cf92737cca167cbf05b47880757846a8400e.jpg)

![4841e6e1db057beae67c92fca76f32184c3fc2cf4337b113d36657d9f923370e.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/4841e6e1db057beae67c92fca76f32184c3fc2cf4337b113d36657d9f923370e.jpg)

![4ad93330c22b19d63059e80a0ba04f7b8994950c569485195dedd5a01be63300.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/4ad93330c22b19d63059e80a0ba04f7b8994950c569485195dedd5a01be63300.jpg)

![4dad3fb6ece170666563ec2f3a069d97104326b84d20aa30b336c19f64ca191b.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/4dad3fb6ece170666563ec2f3a069d97104326b84d20aa30b336c19f64ca191b.jpg)

![5c0f2c81aba25221e9a1dbfb00ce2d1a798f95697d4991d3c6368b7730e9e9b1.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/5c0f2c81aba25221e9a1dbfb00ce2d1a798f95697d4991d3c6368b7730e9e9b1.jpg)

![702ab361bcef9ee2718b69433297f465412488b241a90004ee627756a024462d.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/702ab361bcef9ee2718b69433297f465412488b241a90004ee627756a024462d.jpg)

![93f1c681e6ab1ce20009fac0bce94b85779e3ea6d81a94e03280ef4caeda10a4.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/93f1c681e6ab1ce20009fac0bce94b85779e3ea6d81a94e03280ef4caeda10a4.jpg)

![c50d6a7203a8d8a52777966d3109969ce47cf12d6b2508ef7f6d766378b25488.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/c50d6a7203a8d8a52777966d3109969ce47cf12d6b2508ef7f6d766378b25488.jpg)

![e5aa43d2322d0a060034806297f204b2cc35ba5cb60aa651cf8c5680e29adce9.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/e5aa43d2322d0a060034806297f204b2cc35ba5cb60aa651cf8c5680e29adce9.jpg)

![e79511762a1b273932c28beb7a7682ee41b767f7acaa2e67b2068935d01c1517.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/e79511762a1b273932c28beb7a7682ee41b767f7acaa2e67b2068935d01c1517.jpg)

![eef01f9ff29417f8139797137147239de248a5cf2d4d16a7109a4dbbb6dde389.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/eef01f9ff29417f8139797137147239de248a5cf2d4d16a7109a4dbbb6dde389.jpg)

![f1989616b77e8e6df6898396586b860ed2aef483d5a8538192787118be5acdd5.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/f1989616b77e8e6df6898396586b860ed2aef483d5a8538192787118be5acdd5.jpg)

![f67cf377d7699f2958904e92fc4890181b2d0a27ab13de516ae93ace64897bdc.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/f67cf377d7699f2958904e92fc4890181b2d0a27ab13de516ae93ace64897bdc.jpg)

![fb7a18c16698a37f65ec43a4816ffcf9772dfb7ae0ed97c55d54089549582f21.jpg](../icml_results/6319_DARC_%20Disagreement-Aware%20Alignment%20via%20Risk-Constrained%20Decoding/tables/fb7a18c16698a37f65ec43a4816ffcf9772dfb7ae0ed97c55d54089549582f21.jpg)

<a id="paper-6268"></a>

## MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations


### Images

![0f41b8617fc3eda089cddbc08f5427d50a32811fb74539791f64322cbd8f4c09.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/0f41b8617fc3eda089cddbc08f5427d50a32811fb74539791f64322cbd8f4c09.jpg)

![227ef2b561c7c2a46dced2e597945a885a7f1f080e01e9a5a8b2d3c79a908841.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/227ef2b561c7c2a46dced2e597945a885a7f1f080e01e9a5a8b2d3c79a908841.jpg)

![2acc0573018f8f9d1af783c26701e9b68ad630bebc445774c4fc03987b2df03c.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/2acc0573018f8f9d1af783c26701e9b68ad630bebc445774c4fc03987b2df03c.jpg)

![2bb603d790790f277671afc2b8d72e60b8da606fdfb2dfaa722168100b25efeb.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/2bb603d790790f277671afc2b8d72e60b8da606fdfb2dfaa722168100b25efeb.jpg)

![2bc10f393ace7da5dc3b28bb199591f6a1e7bab0112af466ed08f1a813cb2928.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/2bc10f393ace7da5dc3b28bb199591f6a1e7bab0112af466ed08f1a813cb2928.jpg)

![2fd9f7cec92a4cfb7d4666144febbc6af544659fa0133610b165c5a83d837e7b.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/2fd9f7cec92a4cfb7d4666144febbc6af544659fa0133610b165c5a83d837e7b.jpg)

![36a790bdced7fc1e5ff9571b0b3695c47091bbdb651b04b4a1e8ce3c0de4ba9b.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/36a790bdced7fc1e5ff9571b0b3695c47091bbdb651b04b4a1e8ce3c0de4ba9b.jpg)

![3805f0efcc59bf7def4e36f5494e14814731bfa96af429d171aece2862c1d22c.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/3805f0efcc59bf7def4e36f5494e14814731bfa96af429d171aece2862c1d22c.jpg)

![54aedf33f890230dcdc40da35f43408f08a5c245840df02512a3c0b626a6a4a0.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/54aedf33f890230dcdc40da35f43408f08a5c245840df02512a3c0b626a6a4a0.jpg)

![55d0712515c2673104085df6c9a7ecdad4cd82f0c133cf4b304e0eae24711e4b.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/55d0712515c2673104085df6c9a7ecdad4cd82f0c133cf4b304e0eae24711e4b.jpg)

![5ddd58c4fa86d2b3f8b603a2b3155436e7e7468f0bc2d9369e8bb57653913de2.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/5ddd58c4fa86d2b3f8b603a2b3155436e7e7468f0bc2d9369e8bb57653913de2.jpg)

![7b8042510b8ae9b2e2f9caf47a4cfe92557f0f206db946137ec78faf92999de2.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/7b8042510b8ae9b2e2f9caf47a4cfe92557f0f206db946137ec78faf92999de2.jpg)

![901f26babdc2fcedc84d11e154a1c031db44d771f440a1a2a2ba6354100be55e.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/901f26babdc2fcedc84d11e154a1c031db44d771f440a1a2a2ba6354100be55e.jpg)

![9240d6c80b1121cb722ddcc3b46919530701b6a88023ddcb89ad9e7344c482e4.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/9240d6c80b1121cb722ddcc3b46919530701b6a88023ddcb89ad9e7344c482e4.jpg)

![9300d7a577a535082e1a2e1d326e20879708e4e33b244d6fe6a825aa6521a001.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/9300d7a577a535082e1a2e1d326e20879708e4e33b244d6fe6a825aa6521a001.jpg)

![942233e6a9746de29171641fd2e4fc1d93aa496e1e8af7a7241c76d6e45ea3ab.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/942233e6a9746de29171641fd2e4fc1d93aa496e1e8af7a7241c76d6e45ea3ab.jpg)

![a88fb6e87c00bbaefe58c76892a99aa9b4fcf057665b2ad12f8caf83d68bda1f.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/a88fb6e87c00bbaefe58c76892a99aa9b4fcf057665b2ad12f8caf83d68bda1f.jpg)

![e7f045e9ef98425832cc734864303f7fe236102dc81c29753a88f93e259d7bf1.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/e7f045e9ef98425832cc734864303f7fe236102dc81c29753a88f93e259d7bf1.jpg)

![ea104cb4527e35a9754b9188c61f4ea4635fcbcff619a96f7caabd62341f212f.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/ea104cb4527e35a9754b9188c61f4ea4635fcbcff619a96f7caabd62341f212f.jpg)

![f2058e5cdce3dd2f5510e3ccf87e3c971f5268b132d81d3556dc4ab75ef28600.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/f2058e5cdce3dd2f5510e3ccf87e3c971f5268b132d81d3556dc4ab75ef28600.jpg)

![fc76b5fb8774fb676d050486c7a13e2e004903cdb3ba9126f2d20368cc42fa2a.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/images/fc76b5fb8774fb676d050486c7a13e2e004903cdb3ba9126f2d20368cc42fa2a.jpg)

### Tables

![067590d80e9c217b4d6d2eff6744090bed1b8baa24451c3bd55ca4ac7b4489bc.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/067590d80e9c217b4d6d2eff6744090bed1b8baa24451c3bd55ca4ac7b4489bc.jpg)

![1a687ee9a2e8d4b00de0b48a94d27d9831e2ab56a3550fd339e87edf732432e0.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/1a687ee9a2e8d4b00de0b48a94d27d9831e2ab56a3550fd339e87edf732432e0.jpg)

![3538432b4af6db7dc708e70e0ff8bc88f396c0b0423c19e1c63ffd54a50a23b9.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/3538432b4af6db7dc708e70e0ff8bc88f396c0b0423c19e1c63ffd54a50a23b9.jpg)

![4b7547f18b5c40b2758f6364d9229e9f230b0d3658406fc8c516bfece1a67370.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/4b7547f18b5c40b2758f6364d9229e9f230b0d3658406fc8c516bfece1a67370.jpg)

![62ab63781da7331385471da4aacb12b92a1d85476bcc98cedf6a8c17228d48e9.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/62ab63781da7331385471da4aacb12b92a1d85476bcc98cedf6a8c17228d48e9.jpg)

![67792da1c88ef72ffd3d36952e51baba76930a7bd9d01bab7d3839bf4a67ffb3.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/67792da1c88ef72ffd3d36952e51baba76930a7bd9d01bab7d3839bf4a67ffb3.jpg)

![9b0f85fe2bfcd1319b1bd203a2d0f73071c71a19c325f657e2a34bb0cf4b604d.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/9b0f85fe2bfcd1319b1bd203a2d0f73071c71a19c325f657e2a34bb0cf4b604d.jpg)

![9e29cc56f97737fe54b79048c15b5e4595adff46dab89d8520375a47c2476ae8.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/9e29cc56f97737fe54b79048c15b5e4595adff46dab89d8520375a47c2476ae8.jpg)

![a3a34a9dd1a4e2f96a6951d02e2cc91a4e2203dec37be5b59073ce3ac0c2b039.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/a3a34a9dd1a4e2f96a6951d02e2cc91a4e2203dec37be5b59073ce3ac0c2b039.jpg)

![e27463ccc2d80e2a50c445f2077e7eb882216f035064277104c9a5f4f5c997cd.jpg](../icml_results/6320_MixFP4_%20Enhancing%20NVFP4%20with%20Adaptive%20FP4_INT4%20Block%20Representations/tables/e27463ccc2d80e2a50c445f2077e7eb882216f035064277104c9a5f4f5c997cd.jpg)

<a id="paper-6269"></a>

## Adaptive Residual-Update Steering for Low-Overhead Hallucination Mitigation in Large Vision-Language Models


### Images

![00e19a46c69061d5c79ff86ade9cb874b895b350590de1d181b424601653852f.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/00e19a46c69061d5c79ff86ade9cb874b895b350590de1d181b424601653852f.jpg)

![07125df647b91ac7761c0acd2b341316c0d7e39b769355c58b4d911c73a88d89.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/07125df647b91ac7761c0acd2b341316c0d7e39b769355c58b4d911c73a88d89.jpg)

![12cd1c8e3d2e0ac8e96f098518a383bb73c79cdad0157dc2ec4cdcf66722bdac.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/12cd1c8e3d2e0ac8e96f098518a383bb73c79cdad0157dc2ec4cdcf66722bdac.jpg)

![1465607dccd4238a3aa612c759293bec732311769ab89811dbcee7e21f528bcd.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/1465607dccd4238a3aa612c759293bec732311769ab89811dbcee7e21f528bcd.jpg)

![306c1c9faa68e703dd6fea33e6a26e094c42e16ca45ee6a22ae1f7a820b5c414.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/306c1c9faa68e703dd6fea33e6a26e094c42e16ca45ee6a22ae1f7a820b5c414.jpg)

![352a2f10533aacbdfec64e19fd25e46392bc085828ab27a53d782dea0b197d9d.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/352a2f10533aacbdfec64e19fd25e46392bc085828ab27a53d782dea0b197d9d.jpg)

![3c2f40c8053c6e1d277009863992a4d88745eb8b1b1e8be45abb8a8b1ecc8555.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/3c2f40c8053c6e1d277009863992a4d88745eb8b1b1e8be45abb8a8b1ecc8555.jpg)

![4bdfa6e59642aaab96e2ebfa3841ec14db6c8104fd9d5e323685f509fff6e405.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/4bdfa6e59642aaab96e2ebfa3841ec14db6c8104fd9d5e323685f509fff6e405.jpg)

![4c12fd566a60d018bbecee2991fc78c19afa26f6091c712592bc1f68845112c8.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/4c12fd566a60d018bbecee2991fc78c19afa26f6091c712592bc1f68845112c8.jpg)

![4c5a104a4c8544a3c144329ce37c46143b2e4758f094a3bd4ce35cfcb0e010ed.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/4c5a104a4c8544a3c144329ce37c46143b2e4758f094a3bd4ce35cfcb0e010ed.jpg)

![75d20162fddfa054c75bbe75637506b35b9194f5f79d40411b3f62b2fbc95357.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/75d20162fddfa054c75bbe75637506b35b9194f5f79d40411b3f62b2fbc95357.jpg)

![760b78064c6620a63e717bcc168bf486f19542edd767c20c94049cfdd9323118.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/760b78064c6620a63e717bcc168bf486f19542edd767c20c94049cfdd9323118.jpg)

![770aee5445ea18484a7e0ad17021b708ef3d11451c2af6b118ce2d4db0fc8dfb.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/770aee5445ea18484a7e0ad17021b708ef3d11451c2af6b118ce2d4db0fc8dfb.jpg)

![78a9dc4a957a77c2a52a25a965ac5cecf6bf6378902bc9a9ee333845769e780d.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/78a9dc4a957a77c2a52a25a965ac5cecf6bf6378902bc9a9ee333845769e780d.jpg)

![7900f01ccfa08a94ca085050052360b1a3b827bc117ebffbc44e640a586b86d5.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/7900f01ccfa08a94ca085050052360b1a3b827bc117ebffbc44e640a586b86d5.jpg)

![8cff1ee9850f6e38cfcaeac3190a6917a05f950a630e189adb5b84b6f5bd6840.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/8cff1ee9850f6e38cfcaeac3190a6917a05f950a630e189adb5b84b6f5bd6840.jpg)

![8e64a5c164bace8906b1b5f605a08ffcdd714ed8ec4f0c95aa5c21e4063ffe4a.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/8e64a5c164bace8906b1b5f605a08ffcdd714ed8ec4f0c95aa5c21e4063ffe4a.jpg)

![978949e8eaf15ec0a9a2504b8297a654ff4fa9f958423031ac7d85e682f0928f.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/978949e8eaf15ec0a9a2504b8297a654ff4fa9f958423031ac7d85e682f0928f.jpg)

![9a9763d89dea411c06bbace01f28024026f925ff5b0e49bf7dd73a11d53034e4.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/9a9763d89dea411c06bbace01f28024026f925ff5b0e49bf7dd73a11d53034e4.jpg)

![9cd202689bed3e7baa3b387aef10f72336a975e5982004441d5495f440ab1ebe.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/9cd202689bed3e7baa3b387aef10f72336a975e5982004441d5495f440ab1ebe.jpg)

![ca2ad056fdba5657569b7c161dfad3e13ab5ae09a65c4136afd88867fb52c3d3.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/ca2ad056fdba5657569b7c161dfad3e13ab5ae09a65c4136afd88867fb52c3d3.jpg)

![cb72156e0742e39d1ecc8e1497a32acedcf277182d679a0e8bd5ff7efd9ea6c0.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/cb72156e0742e39d1ecc8e1497a32acedcf277182d679a0e8bd5ff7efd9ea6c0.jpg)

![d0b122d486c56a8e2d499635d07e9fbeb9e335cc2e6a30d9f8bc772c3e7be8aa.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/d0b122d486c56a8e2d499635d07e9fbeb9e335cc2e6a30d9f8bc772c3e7be8aa.jpg)

![d917de40f06405aa9505c266b2b992976440e20d433347a0d2cd1a79f8bfefc0.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/d917de40f06405aa9505c266b2b992976440e20d433347a0d2cd1a79f8bfefc0.jpg)

![f09245962601325074159da84e0abf92efb3c0472a4e70e48675e773f850c204.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/f09245962601325074159da84e0abf92efb3c0472a4e70e48675e773f850c204.jpg)

![faec9b8f5401190e644c11db8750a58cf26ef81735aa3325c2cc9fdb39b02114.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/images/faec9b8f5401190e644c11db8750a58cf26ef81735aa3325c2cc9fdb39b02114.jpg)

### Tables

![27e268d62ff7e2ac7ce1cdbb5732dc3f9ea257c6508136209ed2ba9bb6a6fd60.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/tables/27e268d62ff7e2ac7ce1cdbb5732dc3f9ea257c6508136209ed2ba9bb6a6fd60.jpg)

![5344f2571506e31980b8954eaac291897334c913c06b2fb5fa30da46b3d4ce09.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/tables/5344f2571506e31980b8954eaac291897334c913c06b2fb5fa30da46b3d4ce09.jpg)

![ac42852da41c5920c2ad88f789355e5d86bf2e67027b2fef7c46f087138120f5.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/tables/ac42852da41c5920c2ad88f789355e5d86bf2e67027b2fef7c46f087138120f5.jpg)

![dc281563e647c2c0f0b882d09b209f26343bd2a5e7a288c194cd7ee0819aa4ce.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/tables/dc281563e647c2c0f0b882d09b209f26343bd2a5e7a288c194cd7ee0819aa4ce.jpg)

![e37715414522a99c6e690b51e2e2ce69cada66864799e464ac739f0dcfcda209.jpg](../icml_results/6321_Adaptive%20Residual-Update%20Steering%20for%20Low-Overhead%20Hallucination%20Mitigation%20in%20Large%20Vision-Language/tables/e37715414522a99c6e690b51e2e2ce69cada66864799e464ac739f0dcfcda209.jpg)

<a id="paper-6270"></a>

## History-Bootstrapped Flow Matching for Inverse Boiling Reconstruction

### Images

![1e51f2b059b92d1c7e83e9518a67969ef4effc3056528ca6e1a6941445ef7ae2.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/1e51f2b059b92d1c7e83e9518a67969ef4effc3056528ca6e1a6941445ef7ae2.jpg)

![2e3ff0929b3362ab2d0296d1fe085064a8061397e36a56b509911f3c6e4f3d01.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/2e3ff0929b3362ab2d0296d1fe085064a8061397e36a56b509911f3c6e4f3d01.jpg)

![591d13ac9a4a4c500ac1b289981ccf5654f43f30c39f73daa7db1f9906c3c3ed.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/591d13ac9a4a4c500ac1b289981ccf5654f43f30c39f73daa7db1f9906c3c3ed.jpg)

![6392dcac01952136af6eb19e9a1094f3c6bc40a4ba479e93c2f8da959a06a983.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/6392dcac01952136af6eb19e9a1094f3c6bc40a4ba479e93c2f8da959a06a983.jpg)

![7831a4ed43f4d172ed00d59cabc33ceb857128e158e156124df50b307d4cb21d.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/7831a4ed43f4d172ed00d59cabc33ceb857128e158e156124df50b307d4cb21d.jpg)

![8295d3ee6b5f3b08b744d1f8af2d03972debcd18262888f9744a335655baefd1.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/8295d3ee6b5f3b08b744d1f8af2d03972debcd18262888f9744a335655baefd1.jpg)

![877e0374a23aa51656ec29b606384d2b8e6aa9143e423863bb4e84a5f3504148.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/877e0374a23aa51656ec29b606384d2b8e6aa9143e423863bb4e84a5f3504148.jpg)

![a0374b49a96bdcc76acfbabcdf18a2072f57462f5ce33aa9d721e7f740d18714.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/a0374b49a96bdcc76acfbabcdf18a2072f57462f5ce33aa9d721e7f740d18714.jpg)

![cc4d61eee72d942a8863c25a4fbf7169eca8f764ad8a0f765d396450ae7d45d4.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/cc4d61eee72d942a8863c25a4fbf7169eca8f764ad8a0f765d396450ae7d45d4.jpg)

![d5919df4950c1f65e1eade7304157f2bc408879dd89731f3ae97ba3c04bee66c.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/d5919df4950c1f65e1eade7304157f2bc408879dd89731f3ae97ba3c04bee66c.jpg)

![e4e7e388b53162601fbf915770619dca877084499da1350fb5b777ed41503e93.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/images/e4e7e388b53162601fbf915770619dca877084499da1350fb5b777ed41503e93.jpg)

### Tables

![00d1808aef0433af537667af171e85ea7777e009d102e80f15afc72a6dc88265.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/00d1808aef0433af537667af171e85ea7777e009d102e80f15afc72a6dc88265.jpg)

![0c1807641bd82e098ee6f5f7a89032459e50e2035e41ccb3fa54f4a4f4213382.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/0c1807641bd82e098ee6f5f7a89032459e50e2035e41ccb3fa54f4a4f4213382.jpg)

![1c5646ea61950a3f5dd99ae7a81fb89526994ba6e0ce5b79de7ab2f8cead4c57.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/1c5646ea61950a3f5dd99ae7a81fb89526994ba6e0ce5b79de7ab2f8cead4c57.jpg)

![299eb66806b91609bc72110ee1836baf7f20a2dc06188d1492c56bc0a540eb78.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/299eb66806b91609bc72110ee1836baf7f20a2dc06188d1492c56bc0a540eb78.jpg)

![2f090011783caad18b0127e3054c777eeba283f9b334c3bfd8c46b1bdafb5f07.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/2f090011783caad18b0127e3054c777eeba283f9b334c3bfd8c46b1bdafb5f07.jpg)

![544ab8bd97bdb848015542b1dc889b7d277488a3a664dff5cc4ba319636735c4.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/544ab8bd97bdb848015542b1dc889b7d277488a3a664dff5cc4ba319636735c4.jpg)

![9285d9eea2b51f46c20edfb0cce377084466e89c7872bdd7302b0ae41a00b653.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/9285d9eea2b51f46c20edfb0cce377084466e89c7872bdd7302b0ae41a00b653.jpg)

![a93c2acc3f3f9e9afde4a7d4ad9fd1a4c06420ed09e0929ca26ee7ca9fd64cda.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/a93c2acc3f3f9e9afde4a7d4ad9fd1a4c06420ed09e0929ca26ee7ca9fd64cda.jpg)

![c7f73a15aa3f5b6195f3e7e971c7609ee06b1ee5f578db5c5e2f0dfbcf1ba9f3.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/c7f73a15aa3f5b6195f3e7e971c7609ee06b1ee5f578db5c5e2f0dfbcf1ba9f3.jpg)

![e923b58c8564186d77639ec9a224feb7c723314baa1ebdb817d28b697245ecc2.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/e923b58c8564186d77639ec9a224feb7c723314baa1ebdb817d28b697245ecc2.jpg)

![f5a24a7478e1f8f3bf5537d3c4f6a22c2beccf54fec3560d823bbbe06e4b6e77.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/f5a24a7478e1f8f3bf5537d3c4f6a22c2beccf54fec3560d823bbbe06e4b6e77.jpg)

![f7ac2c92b3036a61ef928839a243e44a85c16e7fa68083f0228a9a8400943fa9.jpg](../icml_results/6323_History-Bootstrapped%20Flow%20Matching%20for%20Inverse%20Boiling%20Reconstruction/tables/f7ac2c92b3036a61ef928839a243e44a85c16e7fa68083f0228a9a8400943fa9.jpg)
