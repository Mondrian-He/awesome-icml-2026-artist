# ICML 2026 Main Conference Papers

**Summary:** 33 papers with extracted content:
- 📊 Total images: 72146
- 📋 Total tables: 65803
- 📄 Total files: 137949

*Note: Equations have been filtered out and are not included.*

---

# ICML 2026 Main Papers - Part 2 of 197

## 目录 (Table of Contents)

1. [Single-Head Attention in High Dimensions: A Theory of Generalization, Weights Spectra, and Scaling Laws](#paper-0034)
2. [Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections](#paper-0035)
3. [Diffract: Spectral View of LLM Domain Adaptation](#paper-0036)
4. [Provable Bounds for the Learnability of Sample-Compressible Families from Noisy Samples](#paper-0037)
5. [Local Mechanisms of Compositional Generalization in Conditional Diffusion](#paper-0038)
6. [Improving the Performance and Learning Stability of Parallelizable RNNs Designed for Ultra-Low Power Applications](#paper-0039)
7. [SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis](#paper-0040)
8. [Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](#paper-0041)
9. [Fast Spectrally Sparse Signal Reconstruction via Jacobi-Preconditioned Gradient Descent](#paper-0042)
10. [Deep Flow Networks](#paper-0043)
11. [Required Spine Optional Limbs: Heterogeneous Federated Learning via Backbone-sharing and Activation-guided Selection](#paper-0044)
12. [Chamaileon: Cross-Context Binder Design with Contextualized Modeling and Mixed Sampling](#paper-0045)
13. [UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture](#paper-0046)
14. [TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation](#paper-0047)
15. [On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance](#paper-0048)
16. [A Unifying View of Variational Generative Wasserstein Flows](#paper-0049)
17. [A Systematic Study of Behavioral Cloning for Scientific Data Annotation](#paper-0050)
18. [Dynamics Reveals Structure: Challenging the Linear Propagation Assumption](#paper-0051)
19. [Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch](#paper-0052)
20. [Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations](#paper-0053)
21. [Bulk-Calibrated Credal Ambiguity Sets: Fast, Tractable Decision Making under Out-of-Sample Contamination](#paper-0054)
22. [CausalGame: Benchmarking Causal Thinking of LLM Agents in Games](#paper-0055)
23. [High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions](#paper-0056)
24. [Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning](#paper-0057)
25. [Information-Theoretic Disentangled Latent Modeling with Conditional Diffusion for Incomplete Multi-View Clustering](#paper-0058)
26. [Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in Large Language Models](#paper-0059)
27. [Reward-free Alignment for Conflicting Objectives](#paper-0060)
28. [Focus and Dilution: The Multi-stage Learning Process of Attention](#paper-0061)
29. [HypoSpace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds](#paper-0062)
30. [Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement](#paper-0063)
31. [TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control](#paper-0064)
32. [Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units](#paper-0065)
33. [Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning](#paper-0066)

---


<a id="paper-0034"></a>

## Single-Head Attention in High Dimensions: A Theory of Generalization, Weights Spectra, and Scaling Laws

### Images

![0eb65273df68d0cd52f8d85ae2f4080dfaffee4b6c52b2e49de12a909c93a716.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/0eb65273df68d0cd52f8d85ae2f4080dfaffee4b6c52b2e49de12a909c93a716.jpg)

![37f4e8a1371b4649448a58ab7e4fe1762cb8aebf900cf8c3a6146afd0315e092.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/37f4e8a1371b4649448a58ab7e4fe1762cb8aebf900cf8c3a6146afd0315e092.jpg)

![579ca2555662f76378c71b0b8616a1fb8ea06f667ce7d241c89abfce2f9b9044.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/579ca2555662f76378c71b0b8616a1fb8ea06f667ce7d241c89abfce2f9b9044.jpg)

![90f6e20734e494d3b9e8699de8916f21b07faca2db55230ce3808d1aacc31b04.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/90f6e20734e494d3b9e8699de8916f21b07faca2db55230ce3808d1aacc31b04.jpg)

![b67adccf75311204bd100fe95d7f198fa2ca5e6f7992bf581b6842a5839db3fd.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/b67adccf75311204bd100fe95d7f198fa2ca5e6f7992bf581b6842a5839db3fd.jpg)

![cde0128bcbfb8003cfa7c1fb199ae3dcb93cd84308480cdc06b959be4b01dc9e.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/cde0128bcbfb8003cfa7c1fb199ae3dcb93cd84308480cdc06b959be4b01dc9e.jpg)

![d463b34c8a3a837e84151cf27f0414a4167376833771cebdb604d99028fcbdb1.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/d463b34c8a3a837e84151cf27f0414a4167376833771cebdb604d99028fcbdb1.jpg)

![ec1b812b3e0bfba3559d8975923cd9389ba2aef2006caac179623ec3138bd617.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/images/ec1b812b3e0bfba3559d8975923cd9389ba2aef2006caac179623ec3138bd617.jpg)

### Tables

![d4e83d725967d13e5627c63b712fe1d18232461ef290e6f5868478fa612316b5.jpg](../icml_results/34_Optimal%20Transport%20under%20Group%20Fairness%20Constraints/tables/d4e83d725967d13e5627c63b712fe1d18232461ef290e6f5868478fa612316b5.jpg)

## Single-Head Attention in High Dimensions: A Theory of Generalization, Weights Spectra, and Scaling Laws


### Images

![1ef03b153ce35111b0d8aa224d42409aaeea72d1865080942eba550f9bd9e5bb.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/1ef03b153ce35111b0d8aa224d42409aaeea72d1865080942eba550f9bd9e5bb.jpg)

![3371b059f3916ca386199064aa92c4893f1f0151e791c0ac6649f9b87341a6bb.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/3371b059f3916ca386199064aa92c4893f1f0151e791c0ac6649f9b87341a6bb.jpg)

![4d6a78018cf0b8ce7abcfacad056ab073c94ac715e3344552609aca6b29239aa.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/4d6a78018cf0b8ce7abcfacad056ab073c94ac715e3344552609aca6b29239aa.jpg)

![4e22b0dec66cf4e6cdbfb9a4ee3117dbd6e78413492d699328d7f94b829d1e52.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/4e22b0dec66cf4e6cdbfb9a4ee3117dbd6e78413492d699328d7f94b829d1e52.jpg)

![7b74fc3429e46d11c629176c0ee73066f8315af64aeb8ada86fd1040e0ff6d03.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/7b74fc3429e46d11c629176c0ee73066f8315af64aeb8ada86fd1040e0ff6d03.jpg)

![9838152dfa45faf2050b93323053f1532e6859bfce2e5bae8347756a189ec237.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/9838152dfa45faf2050b93323053f1532e6859bfce2e5bae8347756a189ec237.jpg)

![a1b71822826ef464e3a96d609f6632d40a4d2b19e1eb6f26ef0773ca18e26601.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/a1b71822826ef464e3a96d609f6632d40a4d2b19e1eb6f26ef0773ca18e26601.jpg)

![bd810f37e7581ef9a87251b3b7bba3838cfdc22256d9b6a9568e6dd4f5b21b88.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/bd810f37e7581ef9a87251b3b7bba3838cfdc22256d9b6a9568e6dd4f5b21b88.jpg)

![c11faea96384f35a3d8f8d6d7d43a28c40b1b0cd16120a706ab9a2c1067eee60.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/c11faea96384f35a3d8f8d6d7d43a28c40b1b0cd16120a706ab9a2c1067eee60.jpg)

![f25b6ea61109d7e58680736f0fd1b3ba427223e773872e720bc5bf8dfa9bb751.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/f25b6ea61109d7e58680736f0fd1b3ba427223e773872e720bc5bf8dfa9bb751.jpg)

![fd200664681e50921e687dca0b797a1e0743f4176f91f61a790c21b520dae655.jpg](../icml_results/35_Single-Head%20Attention%20in%20High%20Dimensions_%20A%20Theory%20of%20Generalization%2C%20Weights%20Spectra%2C%20and%20Scaling%20L/images/fd200664681e50921e687dca0b797a1e0743f4176f91f61a790c21b520dae655.jpg)

<a id="paper-0035"></a>

## Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections


### Images

![0d3b80930c9cdff0853a562d4acf1defb4f8d09576070b929add5d939f25038a.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/0d3b80930c9cdff0853a562d4acf1defb4f8d09576070b929add5d939f25038a.jpg)

![13dcb5f2e2b8568194c42d430bfafacb61078c10558e8d0fc5745da122da6808.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/13dcb5f2e2b8568194c42d430bfafacb61078c10558e8d0fc5745da122da6808.jpg)

![17ab0e92e395b0c50d13f7f0e3f2be8fdd632ef7771484655ecb2f5cfe78ece2.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/17ab0e92e395b0c50d13f7f0e3f2be8fdd632ef7771484655ecb2f5cfe78ece2.jpg)

![1f22a231670a9b97aba5b49952edb896518b1a578e12937e571c24b6bb6a040a.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/1f22a231670a9b97aba5b49952edb896518b1a578e12937e571c24b6bb6a040a.jpg)

![232f4c20dace1d7a656a38c5618706389d03e7e9475b84ed1fe3502c160e7350.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/232f4c20dace1d7a656a38c5618706389d03e7e9475b84ed1fe3502c160e7350.jpg)

![2477edc26cdb803b5cc68a77cd14b9a3c9fe60c8a4ca676a1c8671dab7a386f8.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/2477edc26cdb803b5cc68a77cd14b9a3c9fe60c8a4ca676a1c8671dab7a386f8.jpg)

![26748bd166654180811297c1b8a4db80684a8f61b5cfa820247d78114a8e9cb4.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/26748bd166654180811297c1b8a4db80684a8f61b5cfa820247d78114a8e9cb4.jpg)

![29583c5c51af45d9362232fa8b3b73ce0680c8ae77f61c432aa9f9296ec61a09.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/29583c5c51af45d9362232fa8b3b73ce0680c8ae77f61c432aa9f9296ec61a09.jpg)

![3172a7974066dfc7c3f53c2cb45a61b3c6e91cf5cc96e5b409a3d36a79da1bc0.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/3172a7974066dfc7c3f53c2cb45a61b3c6e91cf5cc96e5b409a3d36a79da1bc0.jpg)

![4f47c525df7ec507ea08a00c15d42d2ef729fe79d4d3d281268c035df6d43e5e.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/4f47c525df7ec507ea08a00c15d42d2ef729fe79d4d3d281268c035df6d43e5e.jpg)

![609d931a1a4f692f81a351c81e958d9894d83450771647f8647b17f782e08f18.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/609d931a1a4f692f81a351c81e958d9894d83450771647f8647b17f782e08f18.jpg)

![6331552455331bdfa473fd816ea18f5edcf6fff973b4b3a07f93221b862c7acf.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/6331552455331bdfa473fd816ea18f5edcf6fff973b4b3a07f93221b862c7acf.jpg)

![69ef19fc83c9be6bd9e45ac6fad2bbd0684e47fcb72048060dfba85bd943113b.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/69ef19fc83c9be6bd9e45ac6fad2bbd0684e47fcb72048060dfba85bd943113b.jpg)

![77e4582d0fde046383c3594ae1f852cc389458dfb407f18aeb263ed37f0d74d7.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/77e4582d0fde046383c3594ae1f852cc389458dfb407f18aeb263ed37f0d74d7.jpg)

![7d2b128696b061a468ffcab57f37b5bb87fff99f8d7725267f9589ab28cd2a05.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/7d2b128696b061a468ffcab57f37b5bb87fff99f8d7725267f9589ab28cd2a05.jpg)

![80a15d090f08a203dce107bed7554b72cd81393adfcf8aaa83b469083033919d.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/80a15d090f08a203dce107bed7554b72cd81393adfcf8aaa83b469083033919d.jpg)

![8449ce59b37dd46e9e7348266a48da87e02e7af2c017d82c1a2fb9deb19a914c.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/8449ce59b37dd46e9e7348266a48da87e02e7af2c017d82c1a2fb9deb19a914c.jpg)

![88388a5ca8ec407dfd2e8e821806b9c0b19b5cca2cfd98f6b5ae3ded18790d1c.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/88388a5ca8ec407dfd2e8e821806b9c0b19b5cca2cfd98f6b5ae3ded18790d1c.jpg)

![8e83e87f0fb0b9e26d567f920c88315e44a243fe8a953b9542957ba421dcfab1.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/8e83e87f0fb0b9e26d567f920c88315e44a243fe8a953b9542957ba421dcfab1.jpg)

![91e71bb7eff30359e043021f7d30f035531c60a9938c556efbb5550140e53fb4.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/91e71bb7eff30359e043021f7d30f035531c60a9938c556efbb5550140e53fb4.jpg)

![981a6b133c245d18578c7a1cbe6b0eb96feaf6127684f61feec58e279f8be171.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/981a6b133c245d18578c7a1cbe6b0eb96feaf6127684f61feec58e279f8be171.jpg)

![9845203fbf678134f4234f0a5da90117f757c93fccc97a6387b3d6e5c81adf7e.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/9845203fbf678134f4234f0a5da90117f757c93fccc97a6387b3d6e5c81adf7e.jpg)

![9a439e9cbecdcda9e724c55660340b39f0f1f826e59dd7c86bb494038bfebbd6.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/9a439e9cbecdcda9e724c55660340b39f0f1f826e59dd7c86bb494038bfebbd6.jpg)

![ae6fd3c33dbe6a4b525a1325b337f986e6cba65db4b5296a36bc2084d807966b.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/ae6fd3c33dbe6a4b525a1325b337f986e6cba65db4b5296a36bc2084d807966b.jpg)

![afa8355bff27dd181eb599f105f431e94a9f970dc137a6f2e6f26605070797d2.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/afa8355bff27dd181eb599f105f431e94a9f970dc137a6f2e6f26605070797d2.jpg)

![b9e73c3329df66033045e107284b77763d627ba80f51510597e313f96f749353.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/b9e73c3329df66033045e107284b77763d627ba80f51510597e313f96f749353.jpg)

![ba0087f50e26ce1bf6c223c9b8c425c470fadba43c657d3dc573daf78863ab86.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/ba0087f50e26ce1bf6c223c9b8c425c470fadba43c657d3dc573daf78863ab86.jpg)

![c5133cd781a70bf8e5371c5c47cb70a1eb9a3d5ccfc8bccd3e5cf888c7c85527.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/c5133cd781a70bf8e5371c5c47cb70a1eb9a3d5ccfc8bccd3e5cf888c7c85527.jpg)

![d58b558748c77dfb64c5e5c57ae5e3059a1c36d71daf2df5cb7ed6000c194542.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/d58b558748c77dfb64c5e5c57ae5e3059a1c36d71daf2df5cb7ed6000c194542.jpg)

![d5c1564f3c8b71cd023c21fd9b9754e7206c60b6ef8510807ea5bcbb164b8ace.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/d5c1564f3c8b71cd023c21fd9b9754e7206c60b6ef8510807ea5bcbb164b8ace.jpg)

![d83432a0dcde7caddc5e196add46d11e6598973e46f5a53434c1d48a07796713.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/d83432a0dcde7caddc5e196add46d11e6598973e46f5a53434c1d48a07796713.jpg)

![ebfb9fb3d395772398e9f891d54823fc7236c8301715b46f68cfc02d8de41df5.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/ebfb9fb3d395772398e9f891d54823fc7236c8301715b46f68cfc02d8de41df5.jpg)

![edb1715b9de8ecf161a1e163de97ac4d1eab8c062b183ebce83d587378acabe1.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/edb1715b9de8ecf161a1e163de97ac4d1eab8c062b183ebce83d587378acabe1.jpg)

![ee4d276f527d5c30d4fd17841b8e4e2f8203a2d2d30392357c3c3ca689970b8d.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/ee4d276f527d5c30d4fd17841b8e4e2f8203a2d2d30392357c3c3ca689970b8d.jpg)

![f28699e449dd82dca954c7377dac15f67c56c8bac11ac97635f6d33db588031e.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/f28699e449dd82dca954c7377dac15f67c56c8bac11ac97635f6d33db588031e.jpg)

![f8af7b2dad6f86bd2456225dc30ef1ff8b56c8a09a0f6ae8da1cbcf3a9bbd50c.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/f8af7b2dad6f86bd2456225dc30ef1ff8b56c8a09a0f6ae8da1cbcf3a9bbd50c.jpg)

![f99ced4fd3530659433d1fcade3aab1d7f71558e3da70e04ded05a5bbe4291a3.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/f99ced4fd3530659433d1fcade3aab1d7f71558e3da70e04ded05a5bbe4291a3.jpg)

![ff1dd9169c38fc78ae629012c9116b0f7432a23e8c6a901663cab1e2a68adf6b.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/images/ff1dd9169c38fc78ae629012c9116b0f7432a23e8c6a901663cab1e2a68adf6b.jpg)

### Tables

![0f756d4af675907c60edf558f90908426f141154f10fe94ae3cd084c9d43b84e.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/0f756d4af675907c60edf558f90908426f141154f10fe94ae3cd084c9d43b84e.jpg)

![134fb0268053e5b6c85ecf29c8464a00f0a5869429eea299ad5c745487eeef2d.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/134fb0268053e5b6c85ecf29c8464a00f0a5869429eea299ad5c745487eeef2d.jpg)

![179918eb43e52a357127616cf114530c9e12fe2be34faf6680edc839d4e8f268.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/179918eb43e52a357127616cf114530c9e12fe2be34faf6680edc839d4e8f268.jpg)

![18ca128f1a4dc30605ff09ff25c1c792e6fb5536e555f08325b6ae3c4a7d52a7.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/18ca128f1a4dc30605ff09ff25c1c792e6fb5536e555f08325b6ae3c4a7d52a7.jpg)

![205e9cfe740b9d2347dfc4ff9cac0e97a75fcadfcd9607bf2e5317b38828db0b.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/205e9cfe740b9d2347dfc4ff9cac0e97a75fcadfcd9607bf2e5317b38828db0b.jpg)

![246d9b4706f242ecc73f7dcb867d67a413a32cc25c4b0a3c5bbac91d6a40fab7.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/246d9b4706f242ecc73f7dcb867d67a413a32cc25c4b0a3c5bbac91d6a40fab7.jpg)

![283784f9a2585e494820d7c8eab8f3ff3454787c7937b3edeccdce9e5e4ca768.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/283784f9a2585e494820d7c8eab8f3ff3454787c7937b3edeccdce9e5e4ca768.jpg)

![29423e46c0a0e7149cac930075b70c81a516ba8b785910130526a8dd8493ff1c.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/29423e46c0a0e7149cac930075b70c81a516ba8b785910130526a8dd8493ff1c.jpg)

![3dfff8ede8abe42d08c4bf99728ab252261fbcde39b3439976d31aa4c5f985ae.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/3dfff8ede8abe42d08c4bf99728ab252261fbcde39b3439976d31aa4c5f985ae.jpg)

![3e00792d3410102367b5fde06326884282c0f6e6959aaeed015630e5e88031ee.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/3e00792d3410102367b5fde06326884282c0f6e6959aaeed015630e5e88031ee.jpg)

![52806e3c504f44e7be1c16ca3b6ccd8a29207597225126b63a068d7de77b8174.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/52806e3c504f44e7be1c16ca3b6ccd8a29207597225126b63a068d7de77b8174.jpg)

![5551f9f7d34271b7cfa2787294895dd77c35c95686b556269ce81e7760080ec8.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/5551f9f7d34271b7cfa2787294895dd77c35c95686b556269ce81e7760080ec8.jpg)

![5d5579b7978de7b284d01c5ce2d21ec94a7e2669c1a7e01c36a2b16eb9f09881.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/5d5579b7978de7b284d01c5ce2d21ec94a7e2669c1a7e01c36a2b16eb9f09881.jpg)

![693c29bbbe6d145831e8a00e18f867692b2820271946407d611084ed30a63911.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/693c29bbbe6d145831e8a00e18f867692b2820271946407d611084ed30a63911.jpg)

![6b9958fc144273a0a24d51678e41f034943c86c0c700127f526e195b517222ab.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/6b9958fc144273a0a24d51678e41f034943c86c0c700127f526e195b517222ab.jpg)

![6c9fb1e623680cb29724f00cce9d9a0cfa8d04521da89f460bad976be2a7963c.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/6c9fb1e623680cb29724f00cce9d9a0cfa8d04521da89f460bad976be2a7963c.jpg)

![6faddd8cb5937fbd6691871af12551146aef540703a9cb5d3001138b0379e1a9.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/6faddd8cb5937fbd6691871af12551146aef540703a9cb5d3001138b0379e1a9.jpg)

![89fdc2566b79111700750556e39528cd15bc6290289e72dda428b3bb954dc5c2.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/89fdc2566b79111700750556e39528cd15bc6290289e72dda428b3bb954dc5c2.jpg)

![8d77d9de44ff5bfe05d8fa83fbf078e769f7bb04e37f4462381a160da06d0b66.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/8d77d9de44ff5bfe05d8fa83fbf078e769f7bb04e37f4462381a160da06d0b66.jpg)

![9741b33808a45fcae265c53e338a8f3b8f4814ea1e16f1c007ae1453c1ac042e.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/9741b33808a45fcae265c53e338a8f3b8f4814ea1e16f1c007ae1453c1ac042e.jpg)

![9b9af68b0ec456aa0e1d56fef91801649c4edfe36b0e6c5fff2e2868dd35fa6b.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/9b9af68b0ec456aa0e1d56fef91801649c4edfe36b0e6c5fff2e2868dd35fa6b.jpg)

![c07fbc6d6a20a98bff2cc5c9b5f22afbaa8c79e7585a2c6484250a9706a5c518.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/c07fbc6d6a20a98bff2cc5c9b5f22afbaa8c79e7585a2c6484250a9706a5c518.jpg)

![c5325d85b8326832395d8f6722dd62bafd5fcc0b7c7b5f5b593a1d38f29dc100.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/c5325d85b8326832395d8f6722dd62bafd5fcc0b7c7b5f5b593a1d38f29dc100.jpg)

![cd1c4626b96662bbe6fb8107314ac351eeea757a9655c4d9808b3c3ebf073bf0.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/cd1c4626b96662bbe6fb8107314ac351eeea757a9655c4d9808b3c3ebf073bf0.jpg)

![d8ac3cce876ebcdc86821a87939c84f73aa9db675b926e2870019161d08128a1.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/d8ac3cce876ebcdc86821a87939c84f73aa9db675b926e2870019161d08128a1.jpg)

![e4c4aab81f411e79a1854c9c35edc617f0c6febdf5a9ec0120dbb2692e06a46f.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/e4c4aab81f411e79a1854c9c35edc617f0c6febdf5a9ec0120dbb2692e06a46f.jpg)

![e53ba33f9fd9b41ffab618f48bd86b908d4edee034b432e924fe5bb649b636c4.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/e53ba33f9fd9b41ffab618f48bd86b908d4edee034b432e924fe5bb649b636c4.jpg)

![eeb80316a0bc67c5ca16096a0997f3b4371b5b8916640c4a235f28a4b25d9805.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/eeb80316a0bc67c5ca16096a0997f3b4371b5b8916640c4a235f28a4b25d9805.jpg)

![fa9a6492211fa42d3d8a7292a0b51064fa3264b0ddc357a2ba5a23cff2008bed.jpg](../icml_results/36_Strategic%20Navigation%20or%20Stochastic%20Search_%20How%20Agents%20and%20Humans%20Reason%20Over%20Document%20Collections/tables/fa9a6492211fa42d3d8a7292a0b51064fa3264b0ddc357a2ba5a23cff2008bed.jpg)

<a id="paper-0036"></a>

## Diffract: Spectral View of LLM Domain Adaptation


### Images

![1ec95c888e70272ca7ffedc6062a382af535e4ccd7a1e23b7fc0c735538ff32a.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/1ec95c888e70272ca7ffedc6062a382af535e4ccd7a1e23b7fc0c735538ff32a.jpg)

![278e6cb312c687b25d3f448ec6e02c6eda4b76962d58d012ad2dfe62aaabb65a.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/278e6cb312c687b25d3f448ec6e02c6eda4b76962d58d012ad2dfe62aaabb65a.jpg)

![31dc30cb18ebac30da20dccb09d5443eeb59fe4771bf68439cd2bbc630aa383a.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/31dc30cb18ebac30da20dccb09d5443eeb59fe4771bf68439cd2bbc630aa383a.jpg)

![3e5870756a1d05b7e559bc3523b1e6cbb86a3a4fc097afbd6e5c294343c93dc3.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/3e5870756a1d05b7e559bc3523b1e6cbb86a3a4fc097afbd6e5c294343c93dc3.jpg)

![4b793dfc24cd583b23547c8af9acaaa045c127a2479d57acbad7a47b38dacfbb.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/4b793dfc24cd583b23547c8af9acaaa045c127a2479d57acbad7a47b38dacfbb.jpg)

![586ee7a8264a846374a825f478cca52abbfc364f4b242afa206a8152926a0973.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/586ee7a8264a846374a825f478cca52abbfc364f4b242afa206a8152926a0973.jpg)

![5f8cd1aab2e80359b19c50fba7af0a665d6886313f4741f6d7b5e39f21be14d4.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/5f8cd1aab2e80359b19c50fba7af0a665d6886313f4741f6d7b5e39f21be14d4.jpg)

![60a5e1000be0de3380275c4b4b52ddd13e84b9b839fe61117ee8aeb053e9715b.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/60a5e1000be0de3380275c4b4b52ddd13e84b9b839fe61117ee8aeb053e9715b.jpg)

![6551dd23ba5f93eaca9aef29c3f68be341756846a99a7e1411c1c2c0c79b3216.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/6551dd23ba5f93eaca9aef29c3f68be341756846a99a7e1411c1c2c0c79b3216.jpg)

![6942580413283ad93e49c7ec25fb8fb27da69f9105fe9b2ca5f237f9455a3d74.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/6942580413283ad93e49c7ec25fb8fb27da69f9105fe9b2ca5f237f9455a3d74.jpg)

![971c093f26b17f5ffa1e629ac3267006fa41498c101109307339a7e41593b85b.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/971c093f26b17f5ffa1e629ac3267006fa41498c101109307339a7e41593b85b.jpg)

![97ffbad0d346761c039a117d3d8ee1de605d4fea61b70f12bd78b244a03a9250.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/97ffbad0d346761c039a117d3d8ee1de605d4fea61b70f12bd78b244a03a9250.jpg)

![aa73a734e55f2c0da5ad4dd99d7dc1a63fe1f7c7f8e8ee93667618fb5d931a66.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/aa73a734e55f2c0da5ad4dd99d7dc1a63fe1f7c7f8e8ee93667618fb5d931a66.jpg)

![bf74c75d8a15f0b178f5d3ac23800bb173c69d81aceab174ed64f3ef4fa0f4cc.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/bf74c75d8a15f0b178f5d3ac23800bb173c69d81aceab174ed64f3ef4fa0f4cc.jpg)

![c952d302c544462a180a66e2a0c82279dd0ab3f6c1505e89efaacbf5a9d33bd9.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/c952d302c544462a180a66e2a0c82279dd0ab3f6c1505e89efaacbf5a9d33bd9.jpg)

![cb438bbda867a9b3a7ccae671276a66bc8e01b4c812ed441906b664d6f0ca8ee.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/cb438bbda867a9b3a7ccae671276a66bc8e01b4c812ed441906b664d6f0ca8ee.jpg)

![d110373535031d27c07408ea04463f9709666d4b68a6557572ec962c34114a0a.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/d110373535031d27c07408ea04463f9709666d4b68a6557572ec962c34114a0a.jpg)

![df76307b5b764f3ae5c440039765522b869f7298e18a8599b19b28b57a19fbef.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/df76307b5b764f3ae5c440039765522b869f7298e18a8599b19b28b57a19fbef.jpg)

![e0961897a4db9762ae06b184f963148f37f8a288e0b1c82c15917d11b8e76fb7.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/e0961897a4db9762ae06b184f963148f37f8a288e0b1c82c15917d11b8e76fb7.jpg)

![ed54d171c68b45bf3ff8a9972160d70e5305b80d301787e8cfda04491d1c717a.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/ed54d171c68b45bf3ff8a9972160d70e5305b80d301787e8cfda04491d1c717a.jpg)

![ed88806c15f2d255879964a7c6a36f5a94fa1f5dd8ab21e8d33dc446ea09d7b8.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/images/ed88806c15f2d255879964a7c6a36f5a94fa1f5dd8ab21e8d33dc446ea09d7b8.jpg)

### Tables

![26eb2513de434ae0a61cb658c102ed8633f2a50ace9d86f3dafc4fe9470b81e9.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/tables/26eb2513de434ae0a61cb658c102ed8633f2a50ace9d86f3dafc4fe9470b81e9.jpg)

![bf92e72a997fb1cfe2c520f098adc7d5668659862e5cb2bc2d76161e577bb38e.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/tables/bf92e72a997fb1cfe2c520f098adc7d5668659862e5cb2bc2d76161e577bb38e.jpg)

![c57fc1cba0c6c18a93bf2598230b74b4f79d2fef09a847266497c9e07708ca2b.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/tables/c57fc1cba0c6c18a93bf2598230b74b4f79d2fef09a847266497c9e07708ca2b.jpg)

![da783c60cac05d01fb93b1b258defc202b9575e11459dda87493e269d5ba7113.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/tables/da783c60cac05d01fb93b1b258defc202b9575e11459dda87493e269d5ba7113.jpg)

![dbc9607a61777e94751f8ebda9d1b9aca2a35f124ef5af96d2727c90dffe80e3.jpg](../icml_results/37_Diffract_%20Spectral%20View%20of%20LLM%20Domain%20Adaptation/tables/dbc9607a61777e94751f8ebda9d1b9aca2a35f124ef5af96d2727c90dffe80e3.jpg)

<a id="paper-0037"></a>

## Provable Bounds for the Learnability of Sample-Compressible Families from Noisy Samples


### Images

![9354a39fb2b41ed525f8a5f7fc0d836ae482a022e7876eff56607ec35b3243e1.jpg](../icml_results/38_Provable%20Bounds%20for%20the%20Learnability%20of%20Sample-Compressible%20Families%20from%20Noisy%20Samples/images/9354a39fb2b41ed525f8a5f7fc0d836ae482a022e7876eff56607ec35b3243e1.jpg)

![fed4d21aaff35497a3e19c68be52285bbd1f52685e18164ae0cd7e807e100c8e.jpg](../icml_results/38_Provable%20Bounds%20for%20the%20Learnability%20of%20Sample-Compressible%20Families%20from%20Noisy%20Samples/images/fed4d21aaff35497a3e19c68be52285bbd1f52685e18164ae0cd7e807e100c8e.jpg)

<a id="paper-0038"></a>

## Local Mechanisms of Compositional Generalization in Conditional Diffusion


### Images

![1046057e3f8d4a3e9278993b69adc28f526f27f579eb32d62e228fad6b3e645c.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/1046057e3f8d4a3e9278993b69adc28f526f27f579eb32d62e228fad6b3e645c.jpg)

![22fc4556864d942ff6d87a84890fb38912b7bc6e6f46622473413c720503e435.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/22fc4556864d942ff6d87a84890fb38912b7bc6e6f46622473413c720503e435.jpg)

![2e5bcf452bc616a92a85b4ac0096aa3622c239d0210ccfabe122663c9792308a.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/2e5bcf452bc616a92a85b4ac0096aa3622c239d0210ccfabe122663c9792308a.jpg)

![3091b594bf32abf5e51254477bded0d1a31ad8e6d62299124e8c29746d904c50.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/3091b594bf32abf5e51254477bded0d1a31ad8e6d62299124e8c29746d904c50.jpg)

![4a7667a3dd97c969a9d7ee1a0b621513a64fdf31a2422b067b1aed313864f90d.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/4a7667a3dd97c969a9d7ee1a0b621513a64fdf31a2422b067b1aed313864f90d.jpg)

![5e97e022a0bea9c258b37e1b250e742f26d3c43366ed3cc47a7b7181f95f4dcc.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/5e97e022a0bea9c258b37e1b250e742f26d3c43366ed3cc47a7b7181f95f4dcc.jpg)

![60637892c41081c93aa1939f0000bf0f4f565a3c9c21e9503a3a44d48f933721.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/60637892c41081c93aa1939f0000bf0f4f565a3c9c21e9503a3a44d48f933721.jpg)

![675e62d62fc2727cefad38a9a5bc65054616e4f5f4a66750e93c32878c825e3d.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/675e62d62fc2727cefad38a9a5bc65054616e4f5f4a66750e93c32878c825e3d.jpg)

![6827795d4a941f19d6b8c205411171f9fa4e057848391a6c4b01a7279000bdc8.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/6827795d4a941f19d6b8c205411171f9fa4e057848391a6c4b01a7279000bdc8.jpg)

![73151749bb7767ce485679a4eda40f25470620b29553f4b83dcabb9314913560.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/73151749bb7767ce485679a4eda40f25470620b29553f4b83dcabb9314913560.jpg)

![733ff8c0a662a64cb64d91314476e30e1c755599b85f049529f67f01ee32cf15.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/733ff8c0a662a64cb64d91314476e30e1c755599b85f049529f67f01ee32cf15.jpg)

![7390ce23de8b7d62d82b7ba92f549418e352ada4a51436520f71e46fd2b991a5.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/7390ce23de8b7d62d82b7ba92f549418e352ada4a51436520f71e46fd2b991a5.jpg)

![7dfd1cca06b2bb954b27ccc358260cc0fd63b42d00b21b02283a5e79bc6673c2.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/7dfd1cca06b2bb954b27ccc358260cc0fd63b42d00b21b02283a5e79bc6673c2.jpg)

![99ba6a63688c6dd9df902cb8efc6ff66a6eb71e62dce2c0ebb393d384b4cc709.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/99ba6a63688c6dd9df902cb8efc6ff66a6eb71e62dce2c0ebb393d384b4cc709.jpg)

![b9d498cda3c7c6ed00f37e75198a8741ff16be1d082fd3b7ab899e36ffcddd9a.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/b9d498cda3c7c6ed00f37e75198a8741ff16be1d082fd3b7ab899e36ffcddd9a.jpg)

![ba405ceced2a888de0614887152a41429674ee5edf36169156b2d45cc08a37a8.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/ba405ceced2a888de0614887152a41429674ee5edf36169156b2d45cc08a37a8.jpg)

![bcfcc6d4fbb77c15853536007ea7a37216e5493cf809fca6c90c942918497eae.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/bcfcc6d4fbb77c15853536007ea7a37216e5493cf809fca6c90c942918497eae.jpg)

![c7b1d49a5775cd85090ff71776e39ebc218f637b3442443c9be2c04415ffd328.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/c7b1d49a5775cd85090ff71776e39ebc218f637b3442443c9be2c04415ffd328.jpg)

![d18b8191165b4eee142e3e5d4d9cc9d8093ed83e38b7ef80271d8c5eb67d7d30.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/d18b8191165b4eee142e3e5d4d9cc9d8093ed83e38b7ef80271d8c5eb67d7d30.jpg)

![d22ca0cc3fee53de12c293ba930fef33f569b60aed5067188e22180780c0fb28.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/d22ca0cc3fee53de12c293ba930fef33f569b60aed5067188e22180780c0fb28.jpg)

![d3a34ad9d390317308a2010f807106a6e60f25ce79768f2c8525faab2e330db8.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/d3a34ad9d390317308a2010f807106a6e60f25ce79768f2c8525faab2e330db8.jpg)

![eaf0c2665dc692d8135e0014e1ea0fbd5de37f7641f7f8ed9d14e894dd305276.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/eaf0c2665dc692d8135e0014e1ea0fbd5de37f7641f7f8ed9d14e894dd305276.jpg)

![ece156e7c4fe4b27af680ece1c44e321b550645bc8c4f89656122699c33fdb57.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/ece156e7c4fe4b27af680ece1c44e321b550645bc8c4f89656122699c33fdb57.jpg)

![fb8c4a2fb7bba00314739fccd09bf882f903215f4c33fd4ac3d1ddaee189b75f.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/images/fb8c4a2fb7bba00314739fccd09bf882f903215f4c33fd4ac3d1ddaee189b75f.jpg)

### Tables

![8431488fe8d64f59fec4133848ad252b6230c3c51c365e202e2dc5bd06829710.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/tables/8431488fe8d64f59fec4133848ad252b6230c3c51c365e202e2dc5bd06829710.jpg)

![90588a04eb3258f2c04a59835233a130ddacd90cb46b58700472768fb882fbd1.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/tables/90588a04eb3258f2c04a59835233a130ddacd90cb46b58700472768fb882fbd1.jpg)

![f1003a1f38ff5a0a4248586c06b694b8b9cbae0add463482558e1cfbacb0e1f5.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/tables/f1003a1f38ff5a0a4248586c06b694b8b9cbae0add463482558e1cfbacb0e1f5.jpg)

![f3484886ddbead3bbd0a2e8d0ac2cabd2ac893bd3225e3724b7dd23791d5c725.jpg](../icml_results/39_Local%20Mechanisms%20of%20Compositional%20Generalization%20in%20Conditional%20Diffusion/tables/f3484886ddbead3bbd0a2e8d0ac2cabd2ac893bd3225e3724b7dd23791d5c725.jpg)

<a id="paper-0039"></a>

## Improving the Performance and Learning Stability of Parallelizable RNNs Designed for Ultra-Low Power Applications


### Images

![20550b42a36c1604d1cc99b5107a1bb0d2dd22261bb25891ffbbca61bd2e9fef.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/20550b42a36c1604d1cc99b5107a1bb0d2dd22261bb25891ffbbca61bd2e9fef.jpg)

![94c00499f9170a0b9cc518add366fff4c45124dd5c8b7a0c43b943df089d909f.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/94c00499f9170a0b9cc518add366fff4c45124dd5c8b7a0c43b943df089d909f.jpg)

![98a99359ba9833478e9366b6096c4a7287b848253634851d234dde0556977289.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/98a99359ba9833478e9366b6096c4a7287b848253634851d234dde0556977289.jpg)

![bb5c655257fa8918deda832229ff70ab6dd719702f716ac0750d7b858b0b9972.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/bb5c655257fa8918deda832229ff70ab6dd719702f716ac0750d7b858b0b9972.jpg)

![c2d89c56ffda79680b87f6d5b0d499c24dd79a6442635fe01bf34cb18a1054ff.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/c2d89c56ffda79680b87f6d5b0d499c24dd79a6442635fe01bf34cb18a1054ff.jpg)

![ed726883b76fccbdabf0b1bf54f6414bc46014185577dae65e858944f2ed6cc0.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/ed726883b76fccbdabf0b1bf54f6414bc46014185577dae65e858944f2ed6cc0.jpg)

![f1a832bc987c675cdd15d3ad767c85b32bd537bd89148d800cb494a414b7ed11.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/images/f1a832bc987c675cdd15d3ad767c85b32bd537bd89148d800cb494a414b7ed11.jpg)

### Tables

![109a5cfcb194a27ca06357b34bde6c5c3689c3bc0a6ecb1ba5cbeb2c2a1cdb09.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/109a5cfcb194a27ca06357b34bde6c5c3689c3bc0a6ecb1ba5cbeb2c2a1cdb09.jpg)

![126423b7e8c8e61e9c91c3efcb46c3b4c85d22ce75306590ba849c1ff9aadfb4.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/126423b7e8c8e61e9c91c3efcb46c3b4c85d22ce75306590ba849c1ff9aadfb4.jpg)

![2de4aeee95151ac79408602070573c54169169a475007b2df3bc10375f7ba2fb.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/2de4aeee95151ac79408602070573c54169169a475007b2df3bc10375f7ba2fb.jpg)

![3937bfe3f82c629c76dcd7c29a42bf66d99185185cb3ca44dbe42653fd363a51.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/3937bfe3f82c629c76dcd7c29a42bf66d99185185cb3ca44dbe42653fd363a51.jpg)

![46b59e9c8ebd2a3705b77ba7220d8af68504dde8fcf8b5356085a0f822471034.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/46b59e9c8ebd2a3705b77ba7220d8af68504dde8fcf8b5356085a0f822471034.jpg)

![54fbaa6b79f6ffe8ff6913a01ece0573e6b5fe604e2d90f1b06c80aee96befd2.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/54fbaa6b79f6ffe8ff6913a01ece0573e6b5fe604e2d90f1b06c80aee96befd2.jpg)

![56c07e32855dc021acc3f17b1b6dc43f41150e1182835e087d9ff070f8d2e28e.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/56c07e32855dc021acc3f17b1b6dc43f41150e1182835e087d9ff070f8d2e28e.jpg)

![69358c41d2df8f4e8b6379ab4925a31951fc87095c329d3bd53bc68b85f93672.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/69358c41d2df8f4e8b6379ab4925a31951fc87095c329d3bd53bc68b85f93672.jpg)

![964faf4285f393d679209a28dd299f1ae86ac6723e5b05a4ef598b7c3275b876.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/964faf4285f393d679209a28dd299f1ae86ac6723e5b05a4ef598b7c3275b876.jpg)

![9c96f7e6ebe717a941fdde23a839b1a775fddbe7190c2ea2521f8da09cc85521.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/9c96f7e6ebe717a941fdde23a839b1a775fddbe7190c2ea2521f8da09cc85521.jpg)

![aadd1d0775954df69a0a6624b0bc10169825f4b262165945af2dc839ba5362fa.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/aadd1d0775954df69a0a6624b0bc10169825f4b262165945af2dc839ba5362fa.jpg)

![ad507ef9b2dd3eeb6fc6a3a5f9e3e27e7b82092788cb661626da139c864dba15.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/ad507ef9b2dd3eeb6fc6a3a5f9e3e27e7b82092788cb661626da139c864dba15.jpg)

![ad87654bb6ab50ec5305bebbddc887464c9648f9c0093b7408244dde6d8b098e.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/ad87654bb6ab50ec5305bebbddc887464c9648f9c0093b7408244dde6d8b098e.jpg)

![b00a6161239e3e678475d4c9764be8eb1c6711a8d6d10953d908728560bc9f78.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/b00a6161239e3e678475d4c9764be8eb1c6711a8d6d10953d908728560bc9f78.jpg)

![ec5221ff6a935cc449cb65c18447ba9f5b358098833c305babfb9119e41d4538.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/ec5221ff6a935cc449cb65c18447ba9f5b358098833c305babfb9119e41d4538.jpg)

![eea6472a416141246251e4ab861762ea2f49cf8ef00abf8780511db91aeb9757.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/eea6472a416141246251e4ab861762ea2f49cf8ef00abf8780511db91aeb9757.jpg)

![fd29f27902d09ed89f4004b708b791615b0e489b3b0b221406dbbe6f3fd2d4ea.jpg](../icml_results/40_Improving%20the%20Performance%20and%20Learning%20Stability%20of%20Parallelizable%20RNNs%20Designed%20for%20Ultra-Low%20Power/tables/fd29f27902d09ed89f4004b708b791615b0e489b3b0b221406dbbe6f3fd2d4ea.jpg)

<a id="paper-0040"></a>

## SurvDiff: A Diffusion Model for Generating Synthetic Data in Survival Analysis


### Images

![051409cfdcd4d1874ba6585d305bc373ba86e2ba2360c65f2833ef4282577f43.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/051409cfdcd4d1874ba6585d305bc373ba86e2ba2360c65f2833ef4282577f43.jpg)

![0cdec59dfbb91cc9d082d3b844582565b415db3b16274fbf695a06fc363422b9.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/0cdec59dfbb91cc9d082d3b844582565b415db3b16274fbf695a06fc363422b9.jpg)

![0ebe11e666b4260a6b2a621376a02eee7fa8b146db922f7ad1196e052672792e.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/0ebe11e666b4260a6b2a621376a02eee7fa8b146db922f7ad1196e052672792e.jpg)

![1640bc851fe9b34bba0dedd85d0ea03a3008ff43fbc955121860485e2eb075fe.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/1640bc851fe9b34bba0dedd85d0ea03a3008ff43fbc955121860485e2eb075fe.jpg)

![376af497c742256f120ddb19c4ba31a338d3a5de85ed2a90d8679798e45af4a0.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/376af497c742256f120ddb19c4ba31a338d3a5de85ed2a90d8679798e45af4a0.jpg)

![41ee57d85555ed6fa719a62a88f3eecff36991f18e92842560b8d1fb872527e3.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/41ee57d85555ed6fa719a62a88f3eecff36991f18e92842560b8d1fb872527e3.jpg)

![51a5aeb4776015986b3159687af8e652ccd9758b2a9811fc0222bf7ae2352c6c.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/51a5aeb4776015986b3159687af8e652ccd9758b2a9811fc0222bf7ae2352c6c.jpg)

![54bf8c938d0bbd6eccc4698ed3cd19f093ae705f6c718b6841baa4bd73618cfe.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/54bf8c938d0bbd6eccc4698ed3cd19f093ae705f6c718b6841baa4bd73618cfe.jpg)

![589685cd00a28a9147558088c68803949dde7636ad636caefa8551253284108c.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/589685cd00a28a9147558088c68803949dde7636ad636caefa8551253284108c.jpg)

![792e008f85ab0740e9e5238efb2d4094633f02a1a698737308c775da84404f92.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/792e008f85ab0740e9e5238efb2d4094633f02a1a698737308c775da84404f92.jpg)

![7b6fc646ec36a9e2c9cbb232b31f12fea5926005215f3f945bd82d86541b86da.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/7b6fc646ec36a9e2c9cbb232b31f12fea5926005215f3f945bd82d86541b86da.jpg)

![afb57ae6e2fab1839ed667cf5578b86137982d6ab251482615f4ccbfb0c4d357.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/afb57ae6e2fab1839ed667cf5578b86137982d6ab251482615f4ccbfb0c4d357.jpg)

![b53d252bee803448762e8e0f5e518177f285b9c4db588a39498571a7692e0b32.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/b53d252bee803448762e8e0f5e518177f285b9c4db588a39498571a7692e0b32.jpg)

![dbe8759ddb61184a3d7d9a3d753379731d476c1797d16c40f7601e83dd1f56f1.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/dbe8759ddb61184a3d7d9a3d753379731d476c1797d16c40f7601e83dd1f56f1.jpg)

![f7925a891a7de383623d0479d8a8cf24593a7894164185c05a52c072ea807939.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/f7925a891a7de383623d0479d8a8cf24593a7894164185c05a52c072ea807939.jpg)

![f8a5d59a97ef1cf8dab7d029e7d75f0cf03c97f49b9143a668cf84acea83796c.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/images/f8a5d59a97ef1cf8dab7d029e7d75f0cf03c97f49b9143a668cf84acea83796c.jpg)

### Tables

![042dc5f15bcdffa8e5ce290123cf66a0c26409984448da9f7639101e006788a8.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/042dc5f15bcdffa8e5ce290123cf66a0c26409984448da9f7639101e006788a8.jpg)

![21b950aa244078b481c8cca8157f6ef390c9b0cd28c80bd3d49c66661c9ed712.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/21b950aa244078b481c8cca8157f6ef390c9b0cd28c80bd3d49c66661c9ed712.jpg)

![2d9b9e2c0d4a9687e750ca40e3b1d309cb881674e876f863f2a8252cb65f9ba5.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/2d9b9e2c0d4a9687e750ca40e3b1d309cb881674e876f863f2a8252cb65f9ba5.jpg)

![2f5630e4392ac343a8d43444fee47f2ba91fd3e681351951232a9d82e3f809a6.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/2f5630e4392ac343a8d43444fee47f2ba91fd3e681351951232a9d82e3f809a6.jpg)

![4c9e2e69c68aea462672e93347a82e1f2f7d2d014f78edc6bbceaf479a99b09b.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/4c9e2e69c68aea462672e93347a82e1f2f7d2d014f78edc6bbceaf479a99b09b.jpg)

![50d9968a91b905d494997debb3f0efb4b9b902bbb6871c0e6318dc7238fe2fb8.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/50d9968a91b905d494997debb3f0efb4b9b902bbb6871c0e6318dc7238fe2fb8.jpg)

![67ae17d82fea4ce7aeac62887b14bfd9f3b50ed105a67a51b74e31af66e6176d.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/67ae17d82fea4ce7aeac62887b14bfd9f3b50ed105a67a51b74e31af66e6176d.jpg)

![6f0f3a27db0b7d0d0a4664aa54de19f7027ad465ec86982ffd7321f6db70e4ff.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/6f0f3a27db0b7d0d0a4664aa54de19f7027ad465ec86982ffd7321f6db70e4ff.jpg)

![a5a2330600ca00dd970fdbc329b612b65bbc0031f7b422f9fff0f46d7e387f27.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/a5a2330600ca00dd970fdbc329b612b65bbc0031f7b422f9fff0f46d7e387f27.jpg)

![ac3234e4ccb4638bc5363278b0dc10cf14049f9a1d21de764d8f5c880b6b8942.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/ac3234e4ccb4638bc5363278b0dc10cf14049f9a1d21de764d8f5c880b6b8942.jpg)

![cbb30371988d2d605db17c5e225f8b4131a9662d02ca610734f6c1814ae4991e.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/cbb30371988d2d605db17c5e225f8b4131a9662d02ca610734f6c1814ae4991e.jpg)

![e4a3ada0b140f8838f777445bd4722f8caa293df8848f3f099ee83385c92ede6.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/e4a3ada0b140f8838f777445bd4722f8caa293df8848f3f099ee83385c92ede6.jpg)

![ebfb094ead9c148989a7e42ecbb02d3629b9ca97f2ec34a83662cf0b4e82803c.jpg](../icml_results/41_SurvDiff_%20A%20Diffusion%20Model%20for%20Generating%20Synthetic%20Data%20in%20Survival%20Analysis/tables/ebfb094ead9c148989a7e42ecbb02d3629b9ca97f2ec34a83662cf0b4e82803c.jpg)

<a id="paper-0041"></a>

## Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents


### Images

![05d60395f5a35b371addb3e97e8b6b4e1d2039abe386eda473d6292e1cb91e26.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/05d60395f5a35b371addb3e97e8b6b4e1d2039abe386eda473d6292e1cb91e26.jpg)

![06f081c1da5c85a464e43d5abb3f8858936e573d7cdbb97a800160c13b60fee9.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/06f081c1da5c85a464e43d5abb3f8858936e573d7cdbb97a800160c13b60fee9.jpg)

![0c7c675f0de2e6c0b7aaa130f4c02a53d7a34a4565af2ccad0686bc65f91f81c.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/0c7c675f0de2e6c0b7aaa130f4c02a53d7a34a4565af2ccad0686bc65f91f81c.jpg)

![1272ef35e39c2041ff798e4aad59a5bc5f7c7c0d06562113d8d6cadc8ee47da4.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/1272ef35e39c2041ff798e4aad59a5bc5f7c7c0d06562113d8d6cadc8ee47da4.jpg)

![1fec3f466c8c635579e8534b41f13848a62dbd927272023f4a1f772c40e5b04f.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/1fec3f466c8c635579e8534b41f13848a62dbd927272023f4a1f772c40e5b04f.jpg)

![266fbd24a8944c06d1e1b84f1cd9cda4ca6559961a24bdabbec3799ad4eaae60.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/266fbd24a8944c06d1e1b84f1cd9cda4ca6559961a24bdabbec3799ad4eaae60.jpg)

![2b29272275f3d2da0bd2ac1ca899b68037c146aca9b4b9394f1aad9c232082d8.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/2b29272275f3d2da0bd2ac1ca899b68037c146aca9b4b9394f1aad9c232082d8.jpg)

![41d2827798616e6364028bb4fb5e45cf6e2063bc60c78d60043b09e81895a5fb.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/41d2827798616e6364028bb4fb5e45cf6e2063bc60c78d60043b09e81895a5fb.jpg)

![4777c01cb24bb2371bc5486975993e1d77b778aecd1d712e3989efc03f0c7ec2.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/4777c01cb24bb2371bc5486975993e1d77b778aecd1d712e3989efc03f0c7ec2.jpg)

![48d4e947caed36cb9c55f8390a65e743b4dbe7271423c519f8ebe3357c7ead24.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/48d4e947caed36cb9c55f8390a65e743b4dbe7271423c519f8ebe3357c7ead24.jpg)

![4b2e06a0244d7d76350d2266f416ce3291f125a2cea64172daf3a423aaa8aa05.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/4b2e06a0244d7d76350d2266f416ce3291f125a2cea64172daf3a423aaa8aa05.jpg)

![5a6c0df892a854191867d2e6b2db9ec039c502f9ca21d97f88e2bd2b116140d8.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/5a6c0df892a854191867d2e6b2db9ec039c502f9ca21d97f88e2bd2b116140d8.jpg)

![5fd366caf25ef7854e5fdab2e48476759a08d10e37c6936afc3e52871828d76b.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/5fd366caf25ef7854e5fdab2e48476759a08d10e37c6936afc3e52871828d76b.jpg)

![61db90e77af952b106410ab771c41ce65d2b0c3f41f4b14e1dbbe07b44e30faa.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/61db90e77af952b106410ab771c41ce65d2b0c3f41f4b14e1dbbe07b44e30faa.jpg)

![631a69e1a63841490713484efcb8a0aed6c505a390ba083b278f5ad94594fbac.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/631a69e1a63841490713484efcb8a0aed6c505a390ba083b278f5ad94594fbac.jpg)

![656edb39a1b8779c54304a31a7ff79aa9d60e8da91e03cabb0bc4134d3b15b91.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/656edb39a1b8779c54304a31a7ff79aa9d60e8da91e03cabb0bc4134d3b15b91.jpg)

![739017e85330de9d155d5e4d2c2dc11be0b3e14cc02b172e8ad271e46f73d0dc.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/739017e85330de9d155d5e4d2c2dc11be0b3e14cc02b172e8ad271e46f73d0dc.jpg)

![7431291ff36de6b9b17efe9c296eb2bdd815bc50b02446da296cfc8793c51c19.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/7431291ff36de6b9b17efe9c296eb2bdd815bc50b02446da296cfc8793c51c19.jpg)

![74a3a18a64fbca15a785fa3027e77d24ddee060a1c75a6be50d55ecf77752de9.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/74a3a18a64fbca15a785fa3027e77d24ddee060a1c75a6be50d55ecf77752de9.jpg)

![78b65f1f5db9d66eb8b8bfe1345143bc6954d74343e3064f133af9701e5a7812.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/78b65f1f5db9d66eb8b8bfe1345143bc6954d74343e3064f133af9701e5a7812.jpg)

![7dd45e4d82ebca40a2de6aff8cc01d8ec833d4435a73195a0f19af51252b8253.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/7dd45e4d82ebca40a2de6aff8cc01d8ec833d4435a73195a0f19af51252b8253.jpg)

![812ea6a00c022c2a93dfd28e65937337555ef8bad0b8449e2e2728de3725c962.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/812ea6a00c022c2a93dfd28e65937337555ef8bad0b8449e2e2728de3725c962.jpg)

![85810e1e5d29bd028acbed9aee0e828f8fc389158b6194c28e8c881a72063585.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/85810e1e5d29bd028acbed9aee0e828f8fc389158b6194c28e8c881a72063585.jpg)

![8dcb3ff6020de4e0c9448aaa8bbd1e12574dfc8a3816f0d6cfedc78b824bf1b4.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/8dcb3ff6020de4e0c9448aaa8bbd1e12574dfc8a3816f0d6cfedc78b824bf1b4.jpg)

![9180a15a21b5a3d3ca2e764ef88a6c68314cd5d8321a1e7105caa055fcfb6376.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/9180a15a21b5a3d3ca2e764ef88a6c68314cd5d8321a1e7105caa055fcfb6376.jpg)

![92821299e338c5efd4dfe5b4dade693ce243e0060b8874e47c37c996ba78007e.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/92821299e338c5efd4dfe5b4dade693ce243e0060b8874e47c37c996ba78007e.jpg)

![95a4edd10d423d825b076fb5ac1b977fcd255309f682bc6885edbcf3f9ff5200.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/95a4edd10d423d825b076fb5ac1b977fcd255309f682bc6885edbcf3f9ff5200.jpg)

![99bac6c8fdc4adc1479feff1726c459da5fdfb6ccb37c9e9593c0a146b95630f.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/99bac6c8fdc4adc1479feff1726c459da5fdfb6ccb37c9e9593c0a146b95630f.jpg)

![9af20a204cb7e7c6d975011aaa48be262361a65c1458651dd2e5a7a6ab97e715.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/9af20a204cb7e7c6d975011aaa48be262361a65c1458651dd2e5a7a6ab97e715.jpg)

![9d202308670e1fb885db25a9254725ffe188036c442795c41f598e8e37d65121.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/9d202308670e1fb885db25a9254725ffe188036c442795c41f598e8e37d65121.jpg)

![9ffe5b11136c3b43d7316df79e16390ac484f1d3aa6a6d724dd4a0c421c84890.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/9ffe5b11136c3b43d7316df79e16390ac484f1d3aa6a6d724dd4a0c421c84890.jpg)

![a549cf0f1ddd33687fd205449f10de15ae5c8a5dbe8fd20118398048bfdc325d.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/a549cf0f1ddd33687fd205449f10de15ae5c8a5dbe8fd20118398048bfdc325d.jpg)

![a6a9ddbd32a1a2b82fc8b437ccbbb0758f9f4abf075163e457468251ac12dc8e.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/a6a9ddbd32a1a2b82fc8b437ccbbb0758f9f4abf075163e457468251ac12dc8e.jpg)

![a71a7328584dbad7f3af9d5e36d322a351aeaa7ff1e2e1b1e1ef0f40dcea4e9f.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/a71a7328584dbad7f3af9d5e36d322a351aeaa7ff1e2e1b1e1ef0f40dcea4e9f.jpg)

![a9118e78224540053f25a2939a3b2ba8d736ae7fd98f2121228924eb4e6ab64b.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/a9118e78224540053f25a2939a3b2ba8d736ae7fd98f2121228924eb4e6ab64b.jpg)

![aa4189648eb9eb7306fa3e88dd20bf679c54deedebed8cd678a6d9cf0521811e.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/aa4189648eb9eb7306fa3e88dd20bf679c54deedebed8cd678a6d9cf0521811e.jpg)

![bb7df55bbf190b5832dd03f909e1667a9fcd313c9d5df22c7e0398ec28e633e6.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/bb7df55bbf190b5832dd03f909e1667a9fcd313c9d5df22c7e0398ec28e633e6.jpg)

![bdcaa4d8d0bd87e8e964244b4ae2873d51b974098086d5cadb6e74cd30c3c4e6.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/bdcaa4d8d0bd87e8e964244b4ae2873d51b974098086d5cadb6e74cd30c3c4e6.jpg)

![be2f5e9502d7bc18862d68ada207e8a9fa585f11b77aed6272c81e168ea97402.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/be2f5e9502d7bc18862d68ada207e8a9fa585f11b77aed6272c81e168ea97402.jpg)

![c16c74d00e1a00b888e7927767e12b6635c4f837f29b4de49a79429e24692106.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/c16c74d00e1a00b888e7927767e12b6635c4f837f29b4de49a79429e24692106.jpg)

![cc9db6c40060a2275bbb835a091775c20e757e039cb9eec5c7ec61cbfb3eb2ee.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/cc9db6c40060a2275bbb835a091775c20e757e039cb9eec5c7ec61cbfb3eb2ee.jpg)

![d1567b1ce953f35a5e987bdeca8c47e2e69f6c08fbb214fc14536aad0b3824b0.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/d1567b1ce953f35a5e987bdeca8c47e2e69f6c08fbb214fc14536aad0b3824b0.jpg)

![d210a1ae3a58596474c664c2dc0bad9619973510581cefff9468d76dc1638798.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/d210a1ae3a58596474c664c2dc0bad9619973510581cefff9468d76dc1638798.jpg)

![e772e91ec504fdebda84eb12a4362350e2059e1f8949d9dd27a0a8610e046644.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/e772e91ec504fdebda84eb12a4362350e2059e1f8949d9dd27a0a8610e046644.jpg)

![e9a52c2e701ba1813d97bdf3afce26662baf7c345b22e69c7df1070a0a214a4f.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/e9a52c2e701ba1813d97bdf3afce26662baf7c345b22e69c7df1070a0a214a4f.jpg)

![fb2b7f8845bf4e23bce77e9638b7b430c9e9cb02bc3e8292a1f5378eefa76c02.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/images/fb2b7f8845bf4e23bce77e9638b7b430c9e9cb02bc3e8292a1f5378eefa76c02.jpg)

### Tables

![27c7f910b82e785275fea4a84e1a1455a7217ad1b0b88ae4bfbe34fad8ade374.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/27c7f910b82e785275fea4a84e1a1455a7217ad1b0b88ae4bfbe34fad8ade374.jpg)

![2d7a3e28b4095e3c735a82bd61ea20d8bb4e261dcbccd3e6c1bcb1e333204b06.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/2d7a3e28b4095e3c735a82bd61ea20d8bb4e261dcbccd3e6c1bcb1e333204b06.jpg)

![4c2f84eb16f9191385ca19eda9acc75f8f2cf2ed4a1d46b818b47a69c9ba346a.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/4c2f84eb16f9191385ca19eda9acc75f8f2cf2ed4a1d46b818b47a69c9ba346a.jpg)

![558c2ebe59fefd7bbbb8d3a00273c56c73003f6e0dd5e702850cda0be324ab89.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/558c2ebe59fefd7bbbb8d3a00273c56c73003f6e0dd5e702850cda0be324ab89.jpg)

![5c9efaaf195d8e1d97a8193761b6c28e3101d24b1b4ba604890b3c2d3d6def94.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/5c9efaaf195d8e1d97a8193761b6c28e3101d24b1b4ba604890b3c2d3d6def94.jpg)

![7b832e6b6bca86d8444586dda6cec668b34db544f02d2462c85c730b4d912a00.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/7b832e6b6bca86d8444586dda6cec668b34db544f02d2462c85c730b4d912a00.jpg)

![8fff7ae4896424fab88bd5bec400094de0f37ce80c8eb6a4f4380206e222573e.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/8fff7ae4896424fab88bd5bec400094de0f37ce80c8eb6a4f4380206e222573e.jpg)

![9144b7cf2159568289fa52ddb05a3eb4d80c2088b4744f74f7900a90ee0391fd.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/9144b7cf2159568289fa52ddb05a3eb4d80c2088b4744f74f7900a90ee0391fd.jpg)

![9b32134afb86f5f140c0d0e15af2adc10d5a37e8f099c91ae76811237a2a83c9.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/9b32134afb86f5f140c0d0e15af2adc10d5a37e8f099c91ae76811237a2a83c9.jpg)

![a354928a972d076f906a22ff9c1c58015e0a5935b31e61b33420f31390a10977.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/a354928a972d076f906a22ff9c1c58015e0a5935b31e61b33420f31390a10977.jpg)

![ad841b0619699708a72f06889f7b197381d7db8eecfac65f6007cfa4470c22a4.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/ad841b0619699708a72f06889f7b197381d7db8eecfac65f6007cfa4470c22a4.jpg)

![cd2cd0f2c230f912a95d51d23d53822fdeffcafeeda4253fa22a726b6e346054.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/cd2cd0f2c230f912a95d51d23d53822fdeffcafeeda4253fa22a726b6e346054.jpg)

![d08faf265e220f650287da89bd448eca1f78e5f3c1ae701bd7224375a2c128c3.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/d08faf265e220f650287da89bd448eca1f78e5f3c1ae701bd7224375a2c128c3.jpg)

![f32ef3c5bdddf55f6e91f4665cbd55775e4ef308b1780a7e2482acc250106825.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/f32ef3c5bdddf55f6e91f4665cbd55775e4ef308b1780a7e2482acc250106825.jpg)

![f9b1fbbd24eea4c95017793ad92beaa418f18e2c1b0d23e5f22ed26d0145ee79.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/f9b1fbbd24eea4c95017793ad92beaa418f18e2c1b0d23e5f22ed26d0145ee79.jpg)

![fe6d576151db6a6f39e083edc7f7f71d122c838b947c3750c7aa2f295e0082af.jpg](../icml_results/42_Recovering%20Policy-Induced%20Errors_%20Benchmarking%20and%20Trajectory%20Synthesis%20for%20Robust%20GUI%20Agents/tables/fe6d576151db6a6f39e083edc7f7f71d122c838b947c3750c7aa2f295e0082af.jpg)

<a id="paper-0042"></a>

## Fast Spectrally Sparse Signal Reconstruction via Jacobi-Preconditioned Gradient Descent


### Images

![42c068f69db5361b593f8dd4f5fa59744d73c14e637d23f20e365d6653e4947a.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/images/42c068f69db5361b593f8dd4f5fa59744d73c14e637d23f20e365d6653e4947a.jpg)

![55beb10b0556cfc8e1db7316cada7c06024ab1126c82df7e4613ac97d229d6c6.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/images/55beb10b0556cfc8e1db7316cada7c06024ab1126c82df7e4613ac97d229d6c6.jpg)

![7316444d80520632aaed14804d1e8b2406bf769698d05edc5996be1614e2df2a.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/images/7316444d80520632aaed14804d1e8b2406bf769698d05edc5996be1614e2df2a.jpg)

![cb8ea7039e3b8831dd4979dcba91e2c2dafaa6329303792f8d54cae12ed45d1b.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/images/cb8ea7039e3b8831dd4979dcba91e2c2dafaa6329303792f8d54cae12ed45d1b.jpg)

![ee3ba5667f0078ea87fe6b3f1714d1b5553447957ba6795b03bb7e4fdc206bf0.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/images/ee3ba5667f0078ea87fe6b3f1714d1b5553447957ba6795b03bb7e4fdc206bf0.jpg)

### Tables

![8bd6410d2f8e2f7e95cc8f1c0923013c331c14ceea3a38d9eaed1bfd830a7530.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/tables/8bd6410d2f8e2f7e95cc8f1c0923013c331c14ceea3a38d9eaed1bfd830a7530.jpg)

![a4c8c103677cf0dbbddd8284bfb3e38dbcdeb533801c5111cc0a63bdfa1f86e0.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/tables/a4c8c103677cf0dbbddd8284bfb3e38dbcdeb533801c5111cc0a63bdfa1f86e0.jpg)

![f2e8655ff274c798f190317085eb203f1be3bc53e4a1641e6a77df3a3f40d07b.jpg](../icml_results/43_Fast%20Spectrally%20Sparse%20Signal%20Reconstruction%20via%20Jacobi-Preconditioned%20Gradient%20Descent/tables/f2e8655ff274c798f190317085eb203f1be3bc53e4a1641e6a77df3a3f40d07b.jpg)

<a id="paper-0043"></a>

## Deep Flow Networks


### Images

![9ad99a388b0fb0f01af9c749467c955449a6ce28755c4f57d81eedade0260adc.jpg](../icml_results/44_Deep%20Flow%20Networks/images/9ad99a388b0fb0f01af9c749467c955449a6ce28755c4f57d81eedade0260adc.jpg)

### Tables

![11121d886e81dba8bee728760deb01bf7c6e0fddd80b0ad80dd9e774ba9cf031.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/11121d886e81dba8bee728760deb01bf7c6e0fddd80b0ad80dd9e774ba9cf031.jpg)

![5e814797ea2bc359dcfd54c97ce2cf276d2117c48db6fd0a96f1f04f67ef7797.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/5e814797ea2bc359dcfd54c97ce2cf276d2117c48db6fd0a96f1f04f67ef7797.jpg)

![77b515696a5dfe9161d7b349869c3052d1f580474b7417d9afda4501c10b2412.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/77b515696a5dfe9161d7b349869c3052d1f580474b7417d9afda4501c10b2412.jpg)

![7c5f83ddea9b4a132d09145bf2101ac5982636a62bffb8e42584741173e174da.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/7c5f83ddea9b4a132d09145bf2101ac5982636a62bffb8e42584741173e174da.jpg)

![87101744f32e0e934190c35034a2958dbbec9a575aaa9fe8f800eaf3e28bbc7a.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/87101744f32e0e934190c35034a2958dbbec9a575aaa9fe8f800eaf3e28bbc7a.jpg)

![9f587260b771efc70c17c26ad6ec91438f6c271a29a47c339c760f836004995e.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/9f587260b771efc70c17c26ad6ec91438f6c271a29a47c339c760f836004995e.jpg)

![a03c82b8283a95690689e115f8e466ca7cf04c7bca4015c895320127a5050c54.jpg](../icml_results/44_Deep%20Flow%20Networks/tables/a03c82b8283a95690689e115f8e466ca7cf04c7bca4015c895320127a5050c54.jpg)

<a id="paper-0044"></a>

## Required Spine Optional Limbs: Heterogeneous Federated Learning via Backbone-sharing and Activation-guided Selection


### Images

![11f1b11d28460690fc41fedac7e82143a953ece8d1006e02cf606cf0daeae53c.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/11f1b11d28460690fc41fedac7e82143a953ece8d1006e02cf606cf0daeae53c.jpg)

![50f4a0b624fd87b86e70165642abc4bbf97ca7962d8512931d8c4e3d31520dc1.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/50f4a0b624fd87b86e70165642abc4bbf97ca7962d8512931d8c4e3d31520dc1.jpg)

![60561f1e151bbb0696c72467cce3399ab2bf92c4cc0a587e13e5e44b964dfb9a.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/60561f1e151bbb0696c72467cce3399ab2bf92c4cc0a587e13e5e44b964dfb9a.jpg)

![78214857bd3797bc318acbe20fc0b0f42626d4b06089e9f00a3c49369b5b1f92.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/78214857bd3797bc318acbe20fc0b0f42626d4b06089e9f00a3c49369b5b1f92.jpg)

![a2e893b3dc17eed10603838b9169c57b9661f67b229848518fc4ecbbfc70b6f3.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/a2e893b3dc17eed10603838b9169c57b9661f67b229848518fc4ecbbfc70b6f3.jpg)

![baa338e730cc4b7ddd9d917b2235eb99d23b0e902eefba71c4f79696dd537dc4.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/baa338e730cc4b7ddd9d917b2235eb99d23b0e902eefba71c4f79696dd537dc4.jpg)

![bb75e81b98d53e4112f7dd4564d841515a57b3caf9af8cfbe5139d8cb54ee9f6.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/images/bb75e81b98d53e4112f7dd4564d841515a57b3caf9af8cfbe5139d8cb54ee9f6.jpg)

### Tables

![62ed4d322fda2c004b462f1b3eae63382a71ac655e0d762eba91c4f36a73a552.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/tables/62ed4d322fda2c004b462f1b3eae63382a71ac655e0d762eba91c4f36a73a552.jpg)

![b36ae0c17cfb6da2aad7e6b859bbb2d2baf4dc34bf63c7290424bba075522c65.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/tables/b36ae0c17cfb6da2aad7e6b859bbb2d2baf4dc34bf63c7290424bba075522c65.jpg)

![dffb855905404ea4cc6a774a6361e4d6868b4e566e74636cef7984e56a279f3a.jpg](../icml_results/45_Required%20Spine%20Optional%20Limbs_%20Heterogeneous%20Federated%20Learning%20via%20Backbone-sharing%20and%20Activation-/tables/dffb855905404ea4cc6a774a6361e4d6868b4e566e74636cef7984e56a279f3a.jpg)

<a id="paper-0045"></a>

## Chamaileon: Cross-Context Binder Design with Contextualized Modeling and Mixed Sampling


### Images

![076401555b9a373fe7cb0215b26cfa1e66b0c1bb11ed8260bc949f9be633543f.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/076401555b9a373fe7cb0215b26cfa1e66b0c1bb11ed8260bc949f9be633543f.jpg)

![203e6e540d239fb07af90aff1f1f680e99e0476a985d8c63dcb8cae17109bad3.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/203e6e540d239fb07af90aff1f1f680e99e0476a985d8c63dcb8cae17109bad3.jpg)

![2c4eb26f3e8c7d91a44ca11ffa7ce3ee86d4ad681aa404dc7a06b4cb208403ed.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/2c4eb26f3e8c7d91a44ca11ffa7ce3ee86d4ad681aa404dc7a06b4cb208403ed.jpg)

![30b692814c8af4eb3caa53f344621e6c9334cb43ceffb876f2938246778588a7.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/30b692814c8af4eb3caa53f344621e6c9334cb43ceffb876f2938246778588a7.jpg)

![3617804942d6f0f820753b4f21a21faf2289211b5e3fcbfed135bb7d93e02db7.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/3617804942d6f0f820753b4f21a21faf2289211b5e3fcbfed135bb7d93e02db7.jpg)

![7d4f4d6e0fec2f303a75c08cf9d077b155d6b4437d8df10987c9d90a605d3605.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/7d4f4d6e0fec2f303a75c08cf9d077b155d6b4437d8df10987c9d90a605d3605.jpg)

![9f950d55194bd6fc04402c1d0230eb3701b65c1e51cd00220951e9e08413eb33.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/9f950d55194bd6fc04402c1d0230eb3701b65c1e51cd00220951e9e08413eb33.jpg)

![a59ba576cb4cfb53c9c76adfe46ad2cd4b60865894f434a41ed293e6f62ee33f.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/a59ba576cb4cfb53c9c76adfe46ad2cd4b60865894f434a41ed293e6f62ee33f.jpg)

![c0b66d08c46061d8dbb4613d3cdb6634fbb7b92d5ee3513b9d8ec023a0cbc7b3.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/c0b66d08c46061d8dbb4613d3cdb6634fbb7b92d5ee3513b9d8ec023a0cbc7b3.jpg)

![d872a880674ab83054f90146651d51231166f1d483d333e70cec07967f3150b1.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/images/d872a880674ab83054f90146651d51231166f1d483d333e70cec07967f3150b1.jpg)

### Tables

![273fbfe128f15a61e38242d5b967d91e9c5aef380ff16d1282ab260ea9f3d5e9.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/273fbfe128f15a61e38242d5b967d91e9c5aef380ff16d1282ab260ea9f3d5e9.jpg)

![293b532e7a2dd8e1c175911507c6d5fa88f90771cab422a3c83842a8d09fc11b.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/293b532e7a2dd8e1c175911507c6d5fa88f90771cab422a3c83842a8d09fc11b.jpg)

![59a2458e4f5a538082f2d1bb448553b3842823883716d22c198de22af7e9dd69.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/59a2458e4f5a538082f2d1bb448553b3842823883716d22c198de22af7e9dd69.jpg)

![8c119b399fc93729d916de811b7294ea034045876d77e3922a251a2ecc935bc4.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/8c119b399fc93729d916de811b7294ea034045876d77e3922a251a2ecc935bc4.jpg)

![9513e5098473d1cc594a4f22f642e52cc85b6817b65967385c25091c86c218c7.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/9513e5098473d1cc594a4f22f642e52cc85b6817b65967385c25091c86c218c7.jpg)

![b1b40d3cdcce9d3c537a1cb1ec36b6e77ea9798f9b91dac4cfb459a7abed0aa6.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/b1b40d3cdcce9d3c537a1cb1ec36b6e77ea9798f9b91dac4cfb459a7abed0aa6.jpg)

![c7f35fd41ae4c34418a41edcf5391193de8a79a4d5702f0a5e6e06bd8eba000c.jpg](../icml_results/46_Chamaileon_%20Cross-Context%20Binder%20Design%20with%20Contextualized%20Modeling%20and%20Mixed%20Sampling/tables/c7f35fd41ae4c34418a41edcf5391193de8a79a4d5702f0a5e6e06bd8eba000c.jpg)

<a id="paper-0046"></a>

## UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture


### Images

![2fa69dce1425ff1d51c759ac98ddd97fd83393ff749b25939853b3bb9e108e01.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/2fa69dce1425ff1d51c759ac98ddd97fd83393ff749b25939853b3bb9e108e01.jpg)

![3bbec04f3a2732c4a31bf9c31daba11ab4679807fad5198fc9c8486194a6eb52.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/3bbec04f3a2732c4a31bf9c31daba11ab4679807fad5198fc9c8486194a6eb52.jpg)

![608f296c241d1f1d76ab0bc79276a98886a2e197fb0138fa84ee588fb03afb1d.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/608f296c241d1f1d76ab0bc79276a98886a2e197fb0138fa84ee588fb03afb1d.jpg)

![65b562a04bfe9abc1f599c35cc696315ac065002e43741947f146c95788ede8c.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/65b562a04bfe9abc1f599c35cc696315ac065002e43741947f146c95788ede8c.jpg)

![6b4bd06236a34cc1c5cf0a234727e77af21ba45fa7bd7d0ee6abf3632fd80c36.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/6b4bd06236a34cc1c5cf0a234727e77af21ba45fa7bd7d0ee6abf3632fd80c36.jpg)

![7a258198892c61d106fb2d8c499c54103952b68051f8928d0238abc6842dee70.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/7a258198892c61d106fb2d8c499c54103952b68051f8928d0238abc6842dee70.jpg)

![7db45343001f0483a82411f236039a7e279519e7137d6a65b8c72c2cf547ad87.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/7db45343001f0483a82411f236039a7e279519e7137d6a65b8c72c2cf547ad87.jpg)

![8732c2f12c89f587115853d027c7ffdbeed1bed0cc02776be15d169c29c3a438.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/8732c2f12c89f587115853d027c7ffdbeed1bed0cc02776be15d169c29c3a438.jpg)

![8c1421070e4c16697cd576b6c9c64f643ea17c3d9c2c9d54a7500ee1c7c7e71a.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/8c1421070e4c16697cd576b6c9c64f643ea17c3d9c2c9d54a7500ee1c7c7e71a.jpg)

![99dd6ac03feb1cd2bbaada96008c3362febcc38e9e21160eba3900c9428cbe13.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/99dd6ac03feb1cd2bbaada96008c3362febcc38e9e21160eba3900c9428cbe13.jpg)

![9e31d3c0dfe3004a89ca65b1e2cb2048d900059a7baf522a06e9d459bf060b31.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/9e31d3c0dfe3004a89ca65b1e2cb2048d900059a7baf522a06e9d459bf060b31.jpg)

![a00f74b84bc2be900bb735801ade9341e947372d7b56f9a98e531c28349bef47.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/a00f74b84bc2be900bb735801ade9341e947372d7b56f9a98e531c28349bef47.jpg)

![a09ed0d59c78ab10b1a19a0be5fdb5cc7db89ad14d2878678088827f77594a3b.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/a09ed0d59c78ab10b1a19a0be5fdb5cc7db89ad14d2878678088827f77594a3b.jpg)

![b0c53ccd0e4d7cce198c0ddbc28188933448399e925462e1b6868bcf0664ccbd.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/b0c53ccd0e4d7cce198c0ddbc28188933448399e925462e1b6868bcf0664ccbd.jpg)

![b5446745989b21ab8a1f94342bd67adb2c5d0906a3d9aa5e1e796493c1bf70d8.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/b5446745989b21ab8a1f94342bd67adb2c5d0906a3d9aa5e1e796493c1bf70d8.jpg)

![ba113c9d9ab7af5cd5c9ac53485ef18ef8e66c2b1bd5693c7040cbcdc9753cd0.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/ba113c9d9ab7af5cd5c9ac53485ef18ef8e66c2b1bd5693c7040cbcdc9753cd0.jpg)

![c3136b59281fba7538aa19042fcb1584cf741ab68648aef64fba3c792d2100a2.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/images/c3136b59281fba7538aa19042fcb1584cf741ab68648aef64fba3c792d2100a2.jpg)

### Tables

![0e136e381225872c6b087700b9db3e69e7a901a07a7dd4b7595a36ea58a45a6d.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/0e136e381225872c6b087700b9db3e69e7a901a07a7dd4b7595a36ea58a45a6d.jpg)

![130227d97d198a1dfd1117c86a509ebc0085c5e5ec6d82e5442e5ef0be316c63.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/130227d97d198a1dfd1117c86a509ebc0085c5e5ec6d82e5442e5ef0be316c63.jpg)

![221cd7f633aa2cb0079ab12d3e45253be066e0e930cb8fd423c498b9603cb4d2.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/221cd7f633aa2cb0079ab12d3e45253be066e0e930cb8fd423c498b9603cb4d2.jpg)

![24ff8980b3810c78a03c975f2e3b0a7650d7c2cbc5c3b0c1a49ecf94f656d466.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/24ff8980b3810c78a03c975f2e3b0a7650d7c2cbc5c3b0c1a49ecf94f656d466.jpg)

![2be66f700634889cc7f87aecf9cb9e4a071cc50e22c9997d90bacf326affd57f.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/2be66f700634889cc7f87aecf9cb9e4a071cc50e22c9997d90bacf326affd57f.jpg)

![3d8fbdc90719e51ca65d68d1799c09e27310a8fcee2c5063effa6ede49d72ebb.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/3d8fbdc90719e51ca65d68d1799c09e27310a8fcee2c5063effa6ede49d72ebb.jpg)

![5866d0de1cfbc59aee286046a1f1e92b1088f4d09fae678ae9fb0b7af411a006.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/5866d0de1cfbc59aee286046a1f1e92b1088f4d09fae678ae9fb0b7af411a006.jpg)

![76d88e05813c70c95e988e3b36bd2c83d59c606664f1ce4f8a86f051e164666e.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/76d88e05813c70c95e988e3b36bd2c83d59c606664f1ce4f8a86f051e164666e.jpg)

![7c0c57317de81a15dc0cc28c7ec3bae05aa92ec4445158e1d709c98e2acf82c7.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/7c0c57317de81a15dc0cc28c7ec3bae05aa92ec4445158e1d709c98e2acf82c7.jpg)

![7db66656d1f4f37e22d073576c4bd91b2c1fc1b25225f78ba59b4a4b7ea93a48.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/7db66656d1f4f37e22d073576c4bd91b2c1fc1b25225f78ba59b4a4b7ea93a48.jpg)

![92477634d6c4898cbc1d1be5b73b85f5f200352fdacef83d681665e6f839e2a3.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/92477634d6c4898cbc1d1be5b73b85f5f200352fdacef83d681665e6f839e2a3.jpg)

![9343ba8e6194658880d87424decc06713c94f74041c3e41287f11caf44d36388.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/9343ba8e6194658880d87424decc06713c94f74041c3e41287f11caf44d36388.jpg)

![9807fa83ebc9ddb63a477954002e1883ee260d0624894344b0778d53293b31aa.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/9807fa83ebc9ddb63a477954002e1883ee260d0624894344b0778d53293b31aa.jpg)

![b0ffd864c0b3fbcc2d0a759e2f5ab234807d3df001ccde531a9ca7282ebca180.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/b0ffd864c0b3fbcc2d0a759e2f5ab234807d3df001ccde531a9ca7282ebca180.jpg)

![c5b89c9521bb264ba8fcdf81c9304c8eb110248fe8c40a5a3fca7ee3f221c825.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/c5b89c9521bb264ba8fcdf81c9304c8eb110248fe8c40a5a3fca7ee3f221c825.jpg)

![c8cced2b614014b9823a21308c58ba6a537898e63dbaf055cedadc1b3f1fe7a9.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/c8cced2b614014b9823a21308c58ba6a537898e63dbaf055cedadc1b3f1fe7a9.jpg)

![d0eb70e328dfebcadf3538c3e84e8673cbe5316773e0f37a3ae8d7a9686e079b.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/d0eb70e328dfebcadf3538c3e84e8673cbe5316773e0f37a3ae8d7a9686e079b.jpg)

![ebf86e1f211d374fa386185b816d50868541b182f6162d4d8cc29576f791ffd0.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/ebf86e1f211d374fa386185b816d50868541b182f6162d4d8cc29576f791ffd0.jpg)

![fb348171a7a47fee16bd9fe7e1d1debb392d566a87975de68aaa26eb264ac887.jpg](../icml_results/47_UniPercept_%20Towards%20Unified%20Perceptual-Level%20Image%20Understanding%20across%20Aesthetics%2C%20Quality%2C%20Structu/tables/fb348171a7a47fee16bd9fe7e1d1debb392d566a87975de68aaa26eb264ac887.jpg)

<a id="paper-0047"></a>

## TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation


### Images

![1ad6443485df093055fd08e617574839da4e2230aec0eadc9fc545b85743a67e.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/images/1ad6443485df093055fd08e617574839da4e2230aec0eadc9fc545b85743a67e.jpg)

![25c84c389a6e9ffd393afe0938086235e1ec1a6958c89a10400f6326722df4a8.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/images/25c84c389a6e9ffd393afe0938086235e1ec1a6958c89a10400f6326722df4a8.jpg)

![b2ca6326c00149e214b06d5a78b1519e3964dafc6ffbd570c52284fa635f97b1.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/images/b2ca6326c00149e214b06d5a78b1519e3964dafc6ffbd570c52284fa635f97b1.jpg)

![ba940ac997a39c9157f38a1212b59fb5bc2c6b4e6ed378e9bf70814d3f155f38.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/images/ba940ac997a39c9157f38a1212b59fb5bc2c6b4e6ed378e9bf70814d3f155f38.jpg)

![c1e1a2a68296fda8ce8e26749739789d377ca73662c9ba07105e7fe17da9191d.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/images/c1e1a2a68296fda8ce8e26749739789d377ca73662c9ba07105e7fe17da9191d.jpg)

![d60f882959cee6e6110caa8eb68c015e26602514d7663d8a29c88a82dd351804.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/images/d60f882959cee6e6110caa8eb68c015e26602514d7663d8a29c88a82dd351804.jpg)

### Tables

![274ff812ee98bc49440329590e4b9b39db0fee53502beec18168c61e7c9e5e2f.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/tables/274ff812ee98bc49440329590e4b9b39db0fee53502beec18168c61e7c9e5e2f.jpg)

![3d59c798353f1e4d1d0c9954875376edb08966d79a90efd74f29bcbf1e87c6f7.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/tables/3d59c798353f1e4d1d0c9954875376edb08966d79a90efd74f29bcbf1e87c6f7.jpg)

![4002711c0ec29fc446ae732d18b0763d352f9029d3aff1fcb5a818b3c8a12ca9.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/tables/4002711c0ec29fc446ae732d18b0763d352f9029d3aff1fcb5a818b3c8a12ca9.jpg)

![4949215bd005bab3cb07ea3becf8413d96ecb1f6aad257c6351a951a58da48d8.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/tables/4949215bd005bab3cb07ea3becf8413d96ecb1f6aad257c6351a951a58da48d8.jpg)

![4ca6951504d38050832c7c0849d599a42a06b4594efa527d39be9426c820f09b.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/tables/4ca6951504d38050832c7c0849d599a42a06b4594efa527d39be9426c820f09b.jpg)

![c145479a4421ec9a1a8d9d03196526822fb648939c23a5e245547bc39f6546af.jpg](../icml_results/48_TD3B_%20Transition-Directed%20Discrete%20Diffusion%20for%20Allosteric%20Binder%20Generation/tables/c145479a4421ec9a1a8d9d03196526822fb648939c23a5e245547bc39f6546af.jpg)

<a id="paper-0048"></a>

## On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance


### Images

![099e3334303061cf59172602e0951f73cb73c3689837de73ec2af80e7bb2930f.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/images/099e3334303061cf59172602e0951f73cb73c3689837de73ec2af80e7bb2930f.jpg)

![9f6505d84afe6e5ea29cd81912ab118a12e7a64f3fb00f74d40f3e61d3437159.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/images/9f6505d84afe6e5ea29cd81912ab118a12e7a64f3fb00f74d40f3e61d3437159.jpg)

![adcf6d6d274ec89ca2e31f77bb1cc2a154a7cf688d84064707f3d481990bb520.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/images/adcf6d6d274ec89ca2e31f77bb1cc2a154a7cf688d84064707f3d481990bb520.jpg)

![b62261e0f7679173a71244d49832f3d00a5a3ea5cdbfe7a7b5bbe6f7682abb04.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/images/b62261e0f7679173a71244d49832f3d00a5a3ea5cdbfe7a7b5bbe6f7682abb04.jpg)

![cb22c9329437c67130cba80e3f7f9f61bb26adbdcb4304c77b213a673d77df73.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/images/cb22c9329437c67130cba80e3f7f9f61bb26adbdcb4304c77b213a673d77df73.jpg)

### Tables

![07b8291b74dccc72f4061fe5288b5312cbafef9476d08f25739e3b47371c4a40.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/07b8291b74dccc72f4061fe5288b5312cbafef9476d08f25739e3b47371c4a40.jpg)

![14a837784625bb571ee157e4e1ac31658a01a6608d2e696c4676ba1d985f17e1.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/14a837784625bb571ee157e4e1ac31658a01a6608d2e696c4676ba1d985f17e1.jpg)

![195fe583e7fba1ebe367b7bfdf604ab60dca215e3313c647a869bb80eef96e94.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/195fe583e7fba1ebe367b7bfdf604ab60dca215e3313c647a869bb80eef96e94.jpg)

![1affee3abac1e0801a5838c553ee2db86e247d431716eac48f39afd9831d9632.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/1affee3abac1e0801a5838c553ee2db86e247d431716eac48f39afd9831d9632.jpg)

![2adc25cd3e757b3b6f33121c7f4b51a9b46fe095cc380ca98e1abb38e286ab32.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/2adc25cd3e757b3b6f33121c7f4b51a9b46fe095cc380ca98e1abb38e286ab32.jpg)

![3394d037ec8d211f2b351dd376e2f1e070a20562ef12a7412f073171e03ff074.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/3394d037ec8d211f2b351dd376e2f1e070a20562ef12a7412f073171e03ff074.jpg)

![3ea79e69fd602f66b0f1f53edd34b3b56059f9e8e6bd712deba0467501a509f1.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/3ea79e69fd602f66b0f1f53edd34b3b56059f9e8e6bd712deba0467501a509f1.jpg)

![52ef7a338a24132d98eb10e19823a4d058c426663da2cd2e9882fd5cf0751a9f.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/52ef7a338a24132d98eb10e19823a4d058c426663da2cd2e9882fd5cf0751a9f.jpg)

![61496b3c3f03afba92ec51fd54166de8a03892421dbb52d5354a434b5400b8cc.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/61496b3c3f03afba92ec51fd54166de8a03892421dbb52d5354a434b5400b8cc.jpg)

![63c8e0de142c35c5c68445b1bbd2d60cd5e23604a101a5f87a8c6876b813790a.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/63c8e0de142c35c5c68445b1bbd2d60cd5e23604a101a5f87a8c6876b813790a.jpg)

![74789dd58384ab391d2732861ff41e3079b1744a34b84ad685a57dc83208c411.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/74789dd58384ab391d2732861ff41e3079b1744a34b84ad685a57dc83208c411.jpg)

![84167af5a247ec9426ad0666f1759800e41e01be2f5e3c8d936fb59c8dc14ee3.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/84167af5a247ec9426ad0666f1759800e41e01be2f5e3c8d936fb59c8dc14ee3.jpg)

![87ed8b24f84a692ee572442dd31c89516f9001f78d51da8c4e54d820fdacb619.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/87ed8b24f84a692ee572442dd31c89516f9001f78d51da8c4e54d820fdacb619.jpg)

![8f8719b7c0331b7baebb813fbdf00168efe85471347e536d9d5afc98410718ca.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/8f8719b7c0331b7baebb813fbdf00168efe85471347e536d9d5afc98410718ca.jpg)

![993c90afbe88b39ce45d46954553226b3dfb8aed46db0982135e0f3e8ba534f3.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/993c90afbe88b39ce45d46954553226b3dfb8aed46db0982135e0f3e8ba534f3.jpg)

![996b12927bc7d4fa725eca31592c1f5b9861cc1e1369967b7503f1733a2af431.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/996b12927bc7d4fa725eca31592c1f5b9861cc1e1369967b7503f1733a2af431.jpg)

![a318a2b24d3c7e33ed23804db46c6b117ce6a7fdcf258a121b9c043cc41af022.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/a318a2b24d3c7e33ed23804db46c6b117ce6a7fdcf258a121b9c043cc41af022.jpg)

![a6be5b95002efd3e3493c1efe551c5f46fce2586cb5804d78c5c34f3b5d79816.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/a6be5b95002efd3e3493c1efe551c5f46fce2586cb5804d78c5c34f3b5d79816.jpg)

![a91acf08ba5dc25dfb9a2d00cab37e4f135cf7e076e3e88aab4f4b5cbb907669.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/a91acf08ba5dc25dfb9a2d00cab37e4f135cf7e076e3e88aab4f4b5cbb907669.jpg)

![af68e6e6ade33d8fd639dc2c6dc7b232fc71d4fc61cc3855f620a5c8562619e3.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/af68e6e6ade33d8fd639dc2c6dc7b232fc71d4fc61cc3855f620a5c8562619e3.jpg)

![b0f56dae9bb4fe6a8a20c25aa06969cf33bd6649d318d4bc16278bfee628ef9b.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/b0f56dae9bb4fe6a8a20c25aa06969cf33bd6649d318d4bc16278bfee628ef9b.jpg)

![d3626d5cf6d54312cae83bb85ecc2bbe10b3bb2aa972e428094c2267639ca841.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/d3626d5cf6d54312cae83bb85ecc2bbe10b3bb2aa972e428094c2267639ca841.jpg)

![d3663886c140b09d5676b09c12e589292d0811914a4a0808ccbb907bcac7aa1a.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/d3663886c140b09d5676b09c12e589292d0811914a4a0808ccbb907bcac7aa1a.jpg)

![ddbb2681a1865e3cae962c7ed36e6d76eb19b97ddd45f5846e55c84bc94a799b.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/ddbb2681a1865e3cae962c7ed36e6d76eb19b97ddd45f5846e55c84bc94a799b.jpg)

![e38ec32ab33f1e46419690c7b970a564f85cdcaa371b48280b65f78886d7e5d9.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/e38ec32ab33f1e46419690c7b970a564f85cdcaa371b48280b65f78886d7e5d9.jpg)

![f53c9f7dd9fd737e223bc3b1800b8874f9fc0949efe1dfcd3f9c6eb33eca6fb6.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/f53c9f7dd9fd737e223bc3b1800b8874f9fc0949efe1dfcd3f9c6eb33eca6fb6.jpg)

![f7a6fa6d4dcf997ecbfeec6d2e43f202f8d1ddd8254f02bcaff34b0b5b15f279.jpg](../icml_results/49_On%20the%20Limits%20of%20LLM%20Adaptability_%20Impact%20of%20Model-Internalized%20Priors%20on%20Annotation%20Task%20Performanc/tables/f7a6fa6d4dcf997ecbfeec6d2e43f202f8d1ddd8254f02bcaff34b0b5b15f279.jpg)

<a id="paper-0049"></a>

## A Unifying View of Variational Generative Wasserstein Flows


### Images

![18a13032acfa9c7bfe9585f6be1e0a90b906610bd6c4e8914ee985a450e0e2c0.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/18a13032acfa9c7bfe9585f6be1e0a90b906610bd6c4e8914ee985a450e0e2c0.jpg)

![48280d49238ef67ff0b01222dd307598b7af6a1cd6ea3cf7c368e827bc0c2048.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/48280d49238ef67ff0b01222dd307598b7af6a1cd6ea3cf7c368e827bc0c2048.jpg)

![4c817903bb69fc7de0000ddc8fda7382afeacf3b0d1e2a13329cdd1c4b3dd321.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/4c817903bb69fc7de0000ddc8fda7382afeacf3b0d1e2a13329cdd1c4b3dd321.jpg)

![52fb0cd7ba2a416e0730a9f8886ec0a864e0dc6c2451f001d3522f4ef95fad2b.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/52fb0cd7ba2a416e0730a9f8886ec0a864e0dc6c2451f001d3522f4ef95fad2b.jpg)

![5d54f9c3152d36c5d6a314515c71e8fd6a6f2d1dd22a0ba06076ca46fc5f2c4b.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/5d54f9c3152d36c5d6a314515c71e8fd6a6f2d1dd22a0ba06076ca46fc5f2c4b.jpg)

![633362fb774a5d62edf0943549b84ee9c555f493f00689082939dadd33a3b793.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/633362fb774a5d62edf0943549b84ee9c555f493f00689082939dadd33a3b793.jpg)

![655f230dbe4934983ae6a5d3a17f0c0a0abc779d4a2b7d51db4715e829ed9052.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/655f230dbe4934983ae6a5d3a17f0c0a0abc779d4a2b7d51db4715e829ed9052.jpg)

![6e563100092f08c85e302b7a0f4ee37ba200c45e4956b0e2173398d3ced99873.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/6e563100092f08c85e302b7a0f4ee37ba200c45e4956b0e2173398d3ced99873.jpg)

![8057a8ff519c4cd4233743098257028538ca4d978048125595ac5dfce54b571f.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/8057a8ff519c4cd4233743098257028538ca4d978048125595ac5dfce54b571f.jpg)

![83cbd55f0446cedee0e2c3f4ac6da9d2b4f7b9343f7f3d3be3722cc1b1d02863.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/83cbd55f0446cedee0e2c3f4ac6da9d2b4f7b9343f7f3d3be3722cc1b1d02863.jpg)

![b96b70be29a570b1d756abf15a98d537e6a1cc1f9c578179ca2c1c6adc7025fb.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/b96b70be29a570b1d756abf15a98d537e6a1cc1f9c578179ca2c1c6adc7025fb.jpg)

![def51155acc21967cf88a9a92d8d6c9c5347c6488c39b84891e3e1c9d9fb155c.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/def51155acc21967cf88a9a92d8d6c9c5347c6488c39b84891e3e1c9d9fb155c.jpg)

![f799f8c4eef242dad7b0568b2a274cad0cf1a62f7bf6aed02a33f4ca763e069c.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/f799f8c4eef242dad7b0568b2a274cad0cf1a62f7bf6aed02a33f4ca763e069c.jpg)

![ff328f9eaa2e359ba58752e2d93e7a5c7c1fb13755fff902325053e69fed9bbb.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/images/ff328f9eaa2e359ba58752e2d93e7a5c7c1fb13755fff902325053e69fed9bbb.jpg)

### Tables

![0ac35464db9aa1119843e06933c769f8b341ff3d45e9dc505005efe52a7e223f.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/0ac35464db9aa1119843e06933c769f8b341ff3d45e9dc505005efe52a7e223f.jpg)

![25cf40e3c7393a547be46e39cb1b66bf8b861a1fd9066c1b50f99415e96f61aa.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/25cf40e3c7393a547be46e39cb1b66bf8b861a1fd9066c1b50f99415e96f61aa.jpg)

![3d5659373b7ce02702594e53668d02734050c208d76a425bce46f53348237acd.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/3d5659373b7ce02702594e53668d02734050c208d76a425bce46f53348237acd.jpg)

![44f53fc10bc7939ea5402a70c8c932e9e8f79db1a94fa6433685587259c4bc02.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/44f53fc10bc7939ea5402a70c8c932e9e8f79db1a94fa6433685587259c4bc02.jpg)

![5c19d20402346252d26d295e844397d68b111181b17c03d3e7580fd390c45fbb.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/5c19d20402346252d26d295e844397d68b111181b17c03d3e7580fd390c45fbb.jpg)

![784b63a58d23f934d8de8b71b185388719cf9f95cd85ad13d778b3913abe9616.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/784b63a58d23f934d8de8b71b185388719cf9f95cd85ad13d778b3913abe9616.jpg)

![bba3566af22318058002728dded99632aca79d15ee1b1a7987fcfa7e9a73cfca.jpg](../icml_results/50_A%20Unifying%20View%20of%20Variational%20Generative%20Wasserstein%20Flows/tables/bba3566af22318058002728dded99632aca79d15ee1b1a7987fcfa7e9a73cfca.jpg)

<a id="paper-0050"></a>

## A Systematic Study of Behavioral Cloning for Scientific Data Annotation


### Images

![075329421df294e6b0d37f80d93a77e2e714d7f1841cfc6a53c02c4aa30c3180.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/075329421df294e6b0d37f80d93a77e2e714d7f1841cfc6a53c02c4aa30c3180.jpg)

![0c80cfba5a5bca134dfe798293dd8873d9eb3da71db935ba12bdb90c8de1f8a7.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/0c80cfba5a5bca134dfe798293dd8873d9eb3da71db935ba12bdb90c8de1f8a7.jpg)

![14e70abfe273a55e6507ae9cf0f5b0b0cd2b521e25fe2c62f39e336e307b04ec.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/14e70abfe273a55e6507ae9cf0f5b0b0cd2b521e25fe2c62f39e336e307b04ec.jpg)

![2af78be334025f67f77e9f9bc7c58be41e71bec8798326df8792d2c760d34563.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/2af78be334025f67f77e9f9bc7c58be41e71bec8798326df8792d2c760d34563.jpg)

![2d13bd257dee337ff9dcce51c20bc0286b7c606e66b170adbc4415fa5ff981c4.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/2d13bd257dee337ff9dcce51c20bc0286b7c606e66b170adbc4415fa5ff981c4.jpg)

![35ea70cf9b6debb81b934831945d557decf1fda95a37b4fbd228671f0cc19a39.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/35ea70cf9b6debb81b934831945d557decf1fda95a37b4fbd228671f0cc19a39.jpg)

![3ea4180dc5ae8392ede86d87d0761e4b3189023d8ad0d2949a480732bbaa56de.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/3ea4180dc5ae8392ede86d87d0761e4b3189023d8ad0d2949a480732bbaa56de.jpg)

![410414455a255f95d64d657ba9c10ac48c0f8349d8abede289dbad53bbea3814.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/410414455a255f95d64d657ba9c10ac48c0f8349d8abede289dbad53bbea3814.jpg)

![43ee07244368e11d4f7a0f401093cf2d815752c172905f3211c2ef8b6dd16a6e.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/43ee07244368e11d4f7a0f401093cf2d815752c172905f3211c2ef8b6dd16a6e.jpg)

![49fe3117f88441f9d598da623add0a49ead46a340768845f3ca2fb7a1cf6ede7.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/49fe3117f88441f9d598da623add0a49ead46a340768845f3ca2fb7a1cf6ede7.jpg)

![4ecdf8a6d0a28f2aabca7d98ee441c6bbedd3514f63ccf0452b1fc3e580838dc.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/4ecdf8a6d0a28f2aabca7d98ee441c6bbedd3514f63ccf0452b1fc3e580838dc.jpg)

![4fc302173a24e1421a7768287d7fcb0a20d325087f89c313d0543ceca4a13856.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/4fc302173a24e1421a7768287d7fcb0a20d325087f89c313d0543ceca4a13856.jpg)

![528b09223499785aea326689066ac91c3284b735580b1f4401baf4f1068dfdcf.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/528b09223499785aea326689066ac91c3284b735580b1f4401baf4f1068dfdcf.jpg)

![5d8ce6219dd289d79a3fb9e22a44baa7cac8fbe98e991e40f92465623868025d.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/5d8ce6219dd289d79a3fb9e22a44baa7cac8fbe98e991e40f92465623868025d.jpg)

![5eb3eb33503a1752a8d693541ec61b5fc149d4f7c310078b1f4b9720f0d792a8.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/5eb3eb33503a1752a8d693541ec61b5fc149d4f7c310078b1f4b9720f0d792a8.jpg)

![626c5a555a72d237801c6fcdde3b1a56770844d8725a285a48a59197df6bc000.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/626c5a555a72d237801c6fcdde3b1a56770844d8725a285a48a59197df6bc000.jpg)

![6298b81bfa118c9a4a0d0e05b9764a586e69219e026b8ab417fed27f539b8a5b.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/6298b81bfa118c9a4a0d0e05b9764a586e69219e026b8ab417fed27f539b8a5b.jpg)

![6e882f2a6ee92b0d08d06fb3045c00bdbd1fc4715c5ed517c3a0bf81a4b695d5.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/6e882f2a6ee92b0d08d06fb3045c00bdbd1fc4715c5ed517c3a0bf81a4b695d5.jpg)

![75c60732e66016d57628fb5529eb1d6d7fecbaa4f9341f1ced5753b7c410c1fa.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/75c60732e66016d57628fb5529eb1d6d7fecbaa4f9341f1ced5753b7c410c1fa.jpg)

![7f745fc95151d9eb7df768801fff9782cef204aef5b7cd45470bdbd2cb46e588.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/7f745fc95151d9eb7df768801fff9782cef204aef5b7cd45470bdbd2cb46e588.jpg)

![8b1811349733f70ea250fc4a9f836ef6291880553aae4993af626e203b1d1f8f.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/8b1811349733f70ea250fc4a9f836ef6291880553aae4993af626e203b1d1f8f.jpg)

![8d31cb12131dafb8b8303799e0f5d04cd4b1856bd64f8a6fa71379bba2aaf6d0.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/8d31cb12131dafb8b8303799e0f5d04cd4b1856bd64f8a6fa71379bba2aaf6d0.jpg)

![8e997eb73b09e955d3541cf7d371fb5b8a339f0fe7158dc4ddb0659bebb25fe1.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/8e997eb73b09e955d3541cf7d371fb5b8a339f0fe7158dc4ddb0659bebb25fe1.jpg)

![92ff2a17504917cfc09232dd2db7aa4c1b407b3f95dd7fbbccb7109bbe47aaff.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/92ff2a17504917cfc09232dd2db7aa4c1b407b3f95dd7fbbccb7109bbe47aaff.jpg)

![93c9a039a72ad3a707b935075e6fc126fba3e9ba577d4e55c8f6cfe723507fdd.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/93c9a039a72ad3a707b935075e6fc126fba3e9ba577d4e55c8f6cfe723507fdd.jpg)

![9acdd8e36d88c4ca98209e1a05f287a1e63d0c1d0252fc733639fe2d3a1cb921.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/9acdd8e36d88c4ca98209e1a05f287a1e63d0c1d0252fc733639fe2d3a1cb921.jpg)

![9d5617d4acc73f1dd605d4d644acd9d423b41a313c6af68a56cd4628cdf2dd4e.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/9d5617d4acc73f1dd605d4d644acd9d423b41a313c6af68a56cd4628cdf2dd4e.jpg)

![a5d1314b4bd122d99264a4bce1476f4fafa47bfd90b09369b98bb045ba7c18cc.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/a5d1314b4bd122d99264a4bce1476f4fafa47bfd90b09369b98bb045ba7c18cc.jpg)

![b1104169965f8edb7b76a04b664336855a66e305850652d684ed15b300b50399.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/b1104169965f8edb7b76a04b664336855a66e305850652d684ed15b300b50399.jpg)

![b5b4c995be515d2ff01f793756fd0c660ec147c5ad7eea1b2d5852ac586a10ea.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/b5b4c995be515d2ff01f793756fd0c660ec147c5ad7eea1b2d5852ac586a10ea.jpg)

![b758bb38986e0d7efc26b2f30e44e6bcd2d612f00cfd332730e29fe10692ff13.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/b758bb38986e0d7efc26b2f30e44e6bcd2d612f00cfd332730e29fe10692ff13.jpg)

![b9eaee8721e74c8c904b2cafc4fc625db683185929dc9a60e1eac1101373c216.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/b9eaee8721e74c8c904b2cafc4fc625db683185929dc9a60e1eac1101373c216.jpg)

![d1e81a6fc704b3326832fa50482227759c221caced62ed224c6efbbc0402dca4.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/d1e81a6fc704b3326832fa50482227759c221caced62ed224c6efbbc0402dca4.jpg)

![d6df734056ea19bd2e24a9d28443b0241101856c0efbe4e5622b4a2613a5e99f.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/d6df734056ea19bd2e24a9d28443b0241101856c0efbe4e5622b4a2613a5e99f.jpg)

![db3d0161520bea1e5ec1fd77caff7d16192fe2b6e4b452cacf1720e72a7070e0.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/db3d0161520bea1e5ec1fd77caff7d16192fe2b6e4b452cacf1720e72a7070e0.jpg)

![dc2e74b8df9323c12f5fb7dfa8ba7460ff5317c6285f5ea65b7b2be016285c05.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/dc2e74b8df9323c12f5fb7dfa8ba7460ff5317c6285f5ea65b7b2be016285c05.jpg)

![dca3d44d93762d43d49d26f236da54b41abf37a742394784f78484ae3449618a.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/dca3d44d93762d43d49d26f236da54b41abf37a742394784f78484ae3449618a.jpg)

![eef86f112e0c53db6c43b8b240ff3485a23b709884d89c4e21ece558e2c10825.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/eef86f112e0c53db6c43b8b240ff3485a23b709884d89c4e21ece558e2c10825.jpg)

![efe5a6a6c2d135eddc392309cd68394778882c4b1f8cd66066a14c0c1c24c82d.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/efe5a6a6c2d135eddc392309cd68394778882c4b1f8cd66066a14c0c1c24c82d.jpg)

![f116b9261d796349052e92ebbbd1b9078186e1b0a57fe3b811c44d4270c9e9d7.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/f116b9261d796349052e92ebbbd1b9078186e1b0a57fe3b811c44d4270c9e9d7.jpg)

![f27311c3b86eb1afd7cd6c37ff66c67906d41629b57c8eaff5deaf083df17576.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/f27311c3b86eb1afd7cd6c37ff66c67906d41629b57c8eaff5deaf083df17576.jpg)

![fa5dd113725b4c83bc63dfb45accc6ae6a774b9436945a1786a01ce24aa3155f.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/fa5dd113725b4c83bc63dfb45accc6ae6a774b9436945a1786a01ce24aa3155f.jpg)

![fd8e3bc0ca34db66913627127da3be8db124777c328a1a48de66c61d48d6bfdd.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/images/fd8e3bc0ca34db66913627127da3be8db124777c328a1a48de66c61d48d6bfdd.jpg)

### Tables

![0b6eab3dea2b01372706f01158c8fcdfd4ab6a4148ac9437f9a25f8480211ae7.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/0b6eab3dea2b01372706f01158c8fcdfd4ab6a4148ac9437f9a25f8480211ae7.jpg)

![0f36121116469712e0a97b4eedad8c36b6d851f6d1e5112fa57d4028af717682.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/0f36121116469712e0a97b4eedad8c36b6d851f6d1e5112fa57d4028af717682.jpg)

![1333e0eb2d5e0f27f3c27944d3d7bb81bb2d87d48eb0dcf967f8d41bb5a274d8.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/1333e0eb2d5e0f27f3c27944d3d7bb81bb2d87d48eb0dcf967f8d41bb5a274d8.jpg)

![1b97946e88838647a4a2099f5f9e5a238b3dd50b46072e2e515deb3d81e7f0fe.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/1b97946e88838647a4a2099f5f9e5a238b3dd50b46072e2e515deb3d81e7f0fe.jpg)

![482d45dcfb36f70304f2b0d897889432fbb538b9340d18d05ed2dc3573cf42ee.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/482d45dcfb36f70304f2b0d897889432fbb538b9340d18d05ed2dc3573cf42ee.jpg)

![496aaf980cd0bd901018f2dc58149104ee4ced31c3558aefd82b0a18a8583c65.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/496aaf980cd0bd901018f2dc58149104ee4ced31c3558aefd82b0a18a8583c65.jpg)

![5c7c05c3de3876a9240c6e7f1481a4fc0cd3a47636022e7cfd0a37c7de83149f.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/5c7c05c3de3876a9240c6e7f1481a4fc0cd3a47636022e7cfd0a37c7de83149f.jpg)

![607b4e7383336a81e6cdf569533c9dcde15f1afe9c41aefdb46ba86eb4511bed.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/607b4e7383336a81e6cdf569533c9dcde15f1afe9c41aefdb46ba86eb4511bed.jpg)

![8db74f2ef8f94c185d0cd39ad7e425279ef259d09911707621e8e4822ed6871b.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/8db74f2ef8f94c185d0cd39ad7e425279ef259d09911707621e8e4822ed6871b.jpg)

![9ec74891f278215edd055475ac08b14f2e6965662d4b51e76c6c6b0f287e537d.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/9ec74891f278215edd055475ac08b14f2e6965662d4b51e76c6c6b0f287e537d.jpg)

![a50a3ad9d72f4698e59f00b3589191dfbece981b68de741e28bec71690085f55.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/a50a3ad9d72f4698e59f00b3589191dfbece981b68de741e28bec71690085f55.jpg)

![a68074ce8ec0856e6858f1a8c61644433c18200b2ee82fbc3db73174bf0372ba.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/a68074ce8ec0856e6858f1a8c61644433c18200b2ee82fbc3db73174bf0372ba.jpg)

![ae41afc91cc5f14ab9402c9b790c3e21e3752e112b42f746b4d1b2e96c828135.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/ae41afc91cc5f14ab9402c9b790c3e21e3752e112b42f746b4d1b2e96c828135.jpg)

![ba5df684eafca1794d68383ab9d2c35bf0d3488986964559f9b34122de038bb2.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/ba5df684eafca1794d68383ab9d2c35bf0d3488986964559f9b34122de038bb2.jpg)

![d02d0b203011657bc8e8d44bd8dae6ce59606fd6f4c3339a1d28aa4dcd181685.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/d02d0b203011657bc8e8d44bd8dae6ce59606fd6f4c3339a1d28aa4dcd181685.jpg)

![d21dae6885688a22e6cda8dc138e061a208cf58cf5764976d6dc3664cadcaf7f.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/d21dae6885688a22e6cda8dc138e061a208cf58cf5764976d6dc3664cadcaf7f.jpg)

![e05646847b3acb8dfa3b0e7b6ded9fb5c814c7e0b8679f8cca9539d63ca050c5.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/e05646847b3acb8dfa3b0e7b6ded9fb5c814c7e0b8679f8cca9539d63ca050c5.jpg)

![e595a62f32921bb63cb856d465b970065216ce37158747cad47c72a768d1c303.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/e595a62f32921bb63cb856d465b970065216ce37158747cad47c72a768d1c303.jpg)

![ef0b96cf77355bf606834299457e478aea118a27473fcd2f1665266b4b357360.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/ef0b96cf77355bf606834299457e478aea118a27473fcd2f1665266b4b357360.jpg)

![f6c1126a960e1dcb8ba42e1a9a82320577be406f131de66acd278a9b6b60994e.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/f6c1126a960e1dcb8ba42e1a9a82320577be406f131de66acd278a9b6b60994e.jpg)

![fc055c1075b311b06f0101a5e0aaacd67819394e831f24cfb5c8305738c93c5a.jpg](../icml_results/51_A%20Systematic%20Study%20of%20Behavioral%20Cloning%20for%20Scientific%20Data%20Annotation/tables/fc055c1075b311b06f0101a5e0aaacd67819394e831f24cfb5c8305738c93c5a.jpg)

<a id="paper-0051"></a>

## Dynamics Reveals Structure: Challenging the Linear Propagation Assumption


### Images

![227512d583c6e323424d50b6349d0ca1c30e41140dddf27e8d4d63e20566206b.jpg](../icml_results/52_Dynamics%20Reveals%20Structure_%20Challenging%20the%20Linear%20Propagation%20Assumption/images/227512d583c6e323424d50b6349d0ca1c30e41140dddf27e8d4d63e20566206b.jpg)

![43f4a3b79ae7bbe02e1ca7b6f49fb9bd82dc14da141958fe85095aa502bfa3cb.jpg](../icml_results/52_Dynamics%20Reveals%20Structure_%20Challenging%20the%20Linear%20Propagation%20Assumption/images/43f4a3b79ae7bbe02e1ca7b6f49fb9bd82dc14da141958fe85095aa502bfa3cb.jpg)

![7c49fbdd689c450cd2b60d821932a8d084cc21677a56b9838a6e651dad186253.jpg](../icml_results/52_Dynamics%20Reveals%20Structure_%20Challenging%20the%20Linear%20Propagation%20Assumption/images/7c49fbdd689c450cd2b60d821932a8d084cc21677a56b9838a6e651dad186253.jpg)

![e4bc335fddfbbd2746a8bca1fbced1587dd0e8d841b5a6c057bccc15396881a4.jpg](../icml_results/52_Dynamics%20Reveals%20Structure_%20Challenging%20the%20Linear%20Propagation%20Assumption/images/e4bc335fddfbbd2746a8bca1fbced1587dd0e8d841b5a6c057bccc15396881a4.jpg)

### Tables

![121c2b6f9d6082e523aea2af5666e9eaa5b92d4c9c546d53f8fb2892547e069a.jpg](../icml_results/52_Dynamics%20Reveals%20Structure_%20Challenging%20the%20Linear%20Propagation%20Assumption/tables/121c2b6f9d6082e523aea2af5666e9eaa5b92d4c9c546d53f8fb2892547e069a.jpg)

<a id="paper-0052"></a>

## Efficiently Training Time-to-First-Spike Spiking Neural Networks from Scratch


### Images

![37aefed0e6ca9aef2874457e8f6b663d5cc15b7a7c3a8ef635a2187718ff4ccd.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/37aefed0e6ca9aef2874457e8f6b663d5cc15b7a7c3a8ef635a2187718ff4ccd.jpg)

![698ed290ae8ab9cc07e74a7fd0e3ae8b5568d683d304bf4b95e0ec67550f3238.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/698ed290ae8ab9cc07e74a7fd0e3ae8b5568d683d304bf4b95e0ec67550f3238.jpg)

![9238856870262466533bac2ab4182ff073d3430a292235638a574f4c260655ef.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/9238856870262466533bac2ab4182ff073d3430a292235638a574f4c260655ef.jpg)

![a80531b4222e3a7fe1fd93f2775d5cf62a29f83ae781a5513121565cbc336f35.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/a80531b4222e3a7fe1fd93f2775d5cf62a29f83ae781a5513121565cbc336f35.jpg)

![a912a713ae666323cf27b33338d1c11ebc4ae76f5ccde8ea9853852d36a52fb9.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/a912a713ae666323cf27b33338d1c11ebc4ae76f5ccde8ea9853852d36a52fb9.jpg)

![b537d6d42024e74d490889f4d3ab34089bfda1205f5b9467de73678685662100.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/b537d6d42024e74d490889f4d3ab34089bfda1205f5b9467de73678685662100.jpg)

![d16dd9e24153ffb90fb658b86926f2f8f6b2a537a2b0467d1c7cec6c55fcbbb5.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/d16dd9e24153ffb90fb658b86926f2f8f6b2a537a2b0467d1c7cec6c55fcbbb5.jpg)

![e0da081740eebe9c3638d76f4a89c79e3286255bf448bf18013662e2b08205c5.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/e0da081740eebe9c3638d76f4a89c79e3286255bf448bf18013662e2b08205c5.jpg)

![fc50779b605d74be63757bd59a41c438d996721ca8eed45f1b382174e9d0eaa3.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/images/fc50779b605d74be63757bd59a41c438d996721ca8eed45f1b382174e9d0eaa3.jpg)

### Tables

![22854d141530136c6bdc0b050d5bd4ec7d5d4cba9b1c0291099f81e003fe5d06.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/22854d141530136c6bdc0b050d5bd4ec7d5d4cba9b1c0291099f81e003fe5d06.jpg)

![2bf94620489c4b7d7c63e170dbb4067879c1c5fb96e52826c69e2fc49cc4743d.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/2bf94620489c4b7d7c63e170dbb4067879c1c5fb96e52826c69e2fc49cc4743d.jpg)

![382756abf539bc36c91dacfb8cc3594fc33105c2e43f7effdb4fd5f45b7f3011.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/382756abf539bc36c91dacfb8cc3594fc33105c2e43f7effdb4fd5f45b7f3011.jpg)

![64925e3b13e60f019c5093e5cd48aa1a1fcca1d51f3fc6de25f604edade90d8b.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/64925e3b13e60f019c5093e5cd48aa1a1fcca1d51f3fc6de25f604edade90d8b.jpg)

![b1e20bad6290d13b6aca125e00a5a472d7c6c8a973450fc6db492d8fdf660fff.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/b1e20bad6290d13b6aca125e00a5a472d7c6c8a973450fc6db492d8fdf660fff.jpg)

![cf7560cf625ce4955f3d1658597a47d0626b677b2a92af1d9eb615461525130e.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/cf7560cf625ce4955f3d1658597a47d0626b677b2a92af1d9eb615461525130e.jpg)

![e715396641f4d9a3140a4fab61bcf159983582808d072def800994df0274caeb.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/e715396641f4d9a3140a4fab61bcf159983582808d072def800994df0274caeb.jpg)

![fdc47b6672d0020872aa78bd29221b11a65a62c18e9c5d7cbe73326a8e7b0f34.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/fdc47b6672d0020872aa78bd29221b11a65a62c18e9c5d7cbe73326a8e7b0f34.jpg)

![ff5baea6dad5ae5e90b5c1b1e9cd75cd360a46ad67e6c679d636a39bb6b73b9a.jpg](../icml_results/53_Efficiently%20Training%20Time-to-First-Spike%20Spiking%20Neural%20Networks%20from%20Scratch/tables/ff5baea6dad5ae5e90b5c1b1e9cd75cd360a46ad67e6c679d636a39bb6b73b9a.jpg)

<a id="paper-0053"></a>

## Train for Truth, Keep the Skills: Binary Retrieval-Augmented Reward Mitigates Hallucinations


### Images

![1420ed9cae125e58c48ffc6acd3039d79202f03d2dd4e2ded3986aeae5d81787.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/1420ed9cae125e58c48ffc6acd3039d79202f03d2dd4e2ded3986aeae5d81787.jpg)

![19a1c7ded582b5af529ce598f385ead5c6f3f461a900dddec5e51d535184cfef.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/19a1c7ded582b5af529ce598f385ead5c6f3f461a900dddec5e51d535184cfef.jpg)

![2aaad174fca5d88be204901b6289993b99451f01d51b659ccd655135937fdbbe.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/2aaad174fca5d88be204901b6289993b99451f01d51b659ccd655135937fdbbe.jpg)

![366342d29ca7dfafef7864c9298d4206b689123003b7ec2c8e269f5c3bc7ad86.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/366342d29ca7dfafef7864c9298d4206b689123003b7ec2c8e269f5c3bc7ad86.jpg)

![b1df09bff5a116a30162dd82a7a12af690d5f8c2e37e880813a3e4d14933252d.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/b1df09bff5a116a30162dd82a7a12af690d5f8c2e37e880813a3e4d14933252d.jpg)

![d47478519792549e700ff650aa5a0e7df9ae8bf7abf3b8dced72022f9b934a1c.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/d47478519792549e700ff650aa5a0e7df9ae8bf7abf3b8dced72022f9b934a1c.jpg)

![dc9e55341f402db7efd4deeb26983a512c8dd8e1f4450d0c664f313a50aa7ce3.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/dc9e55341f402db7efd4deeb26983a512c8dd8e1f4450d0c664f313a50aa7ce3.jpg)

![eafcdd57c0025863f6163651a0577d5e4848c85fcc8df97b3e7116ea03bbb9f8.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/images/eafcdd57c0025863f6163651a0577d5e4848c85fcc8df97b3e7116ea03bbb9f8.jpg)

### Tables

![1e9c756779bd2711d6062088b4449172a06e16d8c85aeb95dc4b56e5a74a9171.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/tables/1e9c756779bd2711d6062088b4449172a06e16d8c85aeb95dc4b56e5a74a9171.jpg)

![44546fd89d1e29f7278b3d3e5115b6fe14d0ffa644f64a13212fb5cbcddd6d17.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/tables/44546fd89d1e29f7278b3d3e5115b6fe14d0ffa644f64a13212fb5cbcddd6d17.jpg)

![450d721ae610a6be5830811b1e790b53851badbb912445007aeff7e26a39a5bf.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/tables/450d721ae610a6be5830811b1e790b53851badbb912445007aeff7e26a39a5bf.jpg)

![6d1a22cd77588c72804ef06fca90dfd81172b8b25d7379f0c82f97157ec36c9e.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/tables/6d1a22cd77588c72804ef06fca90dfd81172b8b25d7379f0c82f97157ec36c9e.jpg)

![af50047ccbaeef24ead5d8a160c23322e2ece104aaf7dee215f5136eaf674525.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/tables/af50047ccbaeef24ead5d8a160c23322e2ece104aaf7dee215f5136eaf674525.jpg)

![f69b9c66f82dbbb4cbc9409c8a666002ca7e59d1c165f64ea449ec48bdd702e1.jpg](../icml_results/54_Train%20for%20Truth%2C%20Keep%20the%20Skills_%20Binary%20Retrieval-Augmented%20Reward%20Mitigates%20Hallucinations/tables/f69b9c66f82dbbb4cbc9409c8a666002ca7e59d1c165f64ea449ec48bdd702e1.jpg)

<a id="paper-0054"></a>

## Bulk-Calibrated Credal Ambiguity Sets: Fast, Tractable Decision Making under Out-of-Sample Contamination


### Images

![0f4a6aea4cf514b67f9d79e3475dd27228df23c7ed8f5d88bee6a31d858232a9.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/0f4a6aea4cf514b67f9d79e3475dd27228df23c7ed8f5d88bee6a31d858232a9.jpg)

![175d043955bb002993ffc2fcb0166649e368a3421d8c61c7563fdede8346655e.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/175d043955bb002993ffc2fcb0166649e368a3421d8c61c7563fdede8346655e.jpg)

![216156097bee63ed22afd077d7af6a58efdcd70c7962e8abc4e6a35671e1d103.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/216156097bee63ed22afd077d7af6a58efdcd70c7962e8abc4e6a35671e1d103.jpg)

![26344c6d2b570c217eb6a13e5e78e9c0cce35f3c6eff0e74765532966d46d1ed.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/26344c6d2b570c217eb6a13e5e78e9c0cce35f3c6eff0e74765532966d46d1ed.jpg)

![3c9b6cbef4a0c390aef6cdf6ebadcc2e0eda062e6fd842db6068a17a6f3af2c4.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/3c9b6cbef4a0c390aef6cdf6ebadcc2e0eda062e6fd842db6068a17a6f3af2c4.jpg)

![41ec6dfe223576745c1c8afaf152e7fc9a5e2b29dcc1a29ba626b2ef0d5d396f.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/41ec6dfe223576745c1c8afaf152e7fc9a5e2b29dcc1a29ba626b2ef0d5d396f.jpg)

![5946a7733e6032246c75694a3bbf907d18f3a43f5e77e7527e404ee72361f7c3.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/5946a7733e6032246c75694a3bbf907d18f3a43f5e77e7527e404ee72361f7c3.jpg)

![6287d473639dc9dbaa86e170064f8b9f10a680dffe816f8347e3eaefc9ce0684.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/6287d473639dc9dbaa86e170064f8b9f10a680dffe816f8347e3eaefc9ce0684.jpg)

![681efd063e01a0663f6e125b91d83fe827ea6ba6d4f15f2b3b3d6c2eb1f871d1.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/681efd063e01a0663f6e125b91d83fe827ea6ba6d4f15f2b3b3d6c2eb1f871d1.jpg)

![854c87e8fb06b515870eac59099b4c21785c0d0fe8e62208cf90c4af1e33bb3f.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/854c87e8fb06b515870eac59099b4c21785c0d0fe8e62208cf90c4af1e33bb3f.jpg)

![c2fd833b4adb58e10f2872c7deda4fe4235aa995c2da7d4ce6ec9cc1553877ef.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/c2fd833b4adb58e10f2872c7deda4fe4235aa995c2da7d4ce6ec9cc1553877ef.jpg)

![ce5d5e3bd8d95711b59770bf9e28d36cc3bf87d28bf260df0597ad74aee9374b.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/ce5d5e3bd8d95711b59770bf9e28d36cc3bf87d28bf260df0597ad74aee9374b.jpg)

![cf63a9df13c1d0dc6dc9e7ed4941889255bb050a58c15ce56a5edd9b750a98eb.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/cf63a9df13c1d0dc6dc9e7ed4941889255bb050a58c15ce56a5edd9b750a98eb.jpg)

![d219e55d7245124bc04b7b6007f0876d06f233a1b0f17d49308d2f3c7bb88bdc.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/d219e55d7245124bc04b7b6007f0876d06f233a1b0f17d49308d2f3c7bb88bdc.jpg)

![d48a0a3597d3565fae37ef9b3079eb5e01d0e76f15092551446ddfefb6b51f6a.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/d48a0a3597d3565fae37ef9b3079eb5e01d0e76f15092551446ddfefb6b51f6a.jpg)

![dee9cfafcfc02f619f2d21ad195c096881f56df189a4942755dc492c0324581f.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/dee9cfafcfc02f619f2d21ad195c096881f56df189a4942755dc492c0324581f.jpg)

![e664812f2a8adbc6ba02d569ac2a7312bbdf3710288c0217075d7867054bd897.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/e664812f2a8adbc6ba02d569ac2a7312bbdf3710288c0217075d7867054bd897.jpg)

![e6cbb616da1d047e53d26fb4eb45839226bb86d7bb72345a2f4243246fc2605b.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/images/e6cbb616da1d047e53d26fb4eb45839226bb86d7bb72345a2f4243246fc2605b.jpg)

### Tables

![04e4c1106f42894c7296c81d88dfe031e3c2365b7bb817962c9486f9f8a9566e.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/04e4c1106f42894c7296c81d88dfe031e3c2365b7bb817962c9486f9f8a9566e.jpg)

![3831cd2aba9a908e041bfcd39a2e805074405ce4a5b28189da619cacdabfeb31.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/3831cd2aba9a908e041bfcd39a2e805074405ce4a5b28189da619cacdabfeb31.jpg)

![48a2bccaf982b1dba564ad116a5191d4d4ba06c09b1abbc980edb6176b6af752.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/48a2bccaf982b1dba564ad116a5191d4d4ba06c09b1abbc980edb6176b6af752.jpg)

![4cd43e5ee9817e1f24577e21318eb20fc29720fc5219c2c0a880d2ae8f447471.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/4cd43e5ee9817e1f24577e21318eb20fc29720fc5219c2c0a880d2ae8f447471.jpg)

![744b5fc2a731f1c4edf77b8fd1e5edfa072ec7bf4c8c9a7a847fdbdf140cff96.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/744b5fc2a731f1c4edf77b8fd1e5edfa072ec7bf4c8c9a7a847fdbdf140cff96.jpg)

![9376be26b7d3b78fedd88c5ca13e3b9024c5b1326a7103f9f11b4171fab3ce8f.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/9376be26b7d3b78fedd88c5ca13e3b9024c5b1326a7103f9f11b4171fab3ce8f.jpg)

![b7391ecc2142ab9a42c987031782f78bbf8f9569d6221d317f5d7d16829a47ad.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/b7391ecc2142ab9a42c987031782f78bbf8f9569d6221d317f5d7d16829a47ad.jpg)

![c193df92cafef2cc9cd3fe661229397173a98b3422bdcd103e65ef38c967282d.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/c193df92cafef2cc9cd3fe661229397173a98b3422bdcd103e65ef38c967282d.jpg)

![dabc59c7c0ca17c5ad94c2b4d2da81d2ef19ae0d41c83cf58833ec0227be63a5.jpg](../icml_results/55_Bulk-Calibrated%20Credal%20Ambiguity%20Sets_%20Fast%2C%20Tractable%20Decision%20Making%20under%20Out-of-Sample%20Contamina/tables/dabc59c7c0ca17c5ad94c2b4d2da81d2ef19ae0d41c83cf58833ec0227be63a5.jpg)

<a id="paper-0055"></a>

## CausalGame: Benchmarking Causal Thinking of LLM Agents in Games


### Images

![11aed89b82dc3a425cd34bbf0b03a0cb830ffc06c106fb558f4f866b08feb5a0.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/11aed89b82dc3a425cd34bbf0b03a0cb830ffc06c106fb558f4f866b08feb5a0.jpg)

![178f4b9131d2fb403b87c96c6f17fe1e58d17db592992cacab8637d5eb6a131f.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/178f4b9131d2fb403b87c96c6f17fe1e58d17db592992cacab8637d5eb6a131f.jpg)

![2c87957ba1353033985d9688d490960fa0f042af3fb78e28dee7479961e264cd.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/2c87957ba1353033985d9688d490960fa0f042af3fb78e28dee7479961e264cd.jpg)

![2cea4943ebd4af597ad91da32df1e0fc9f9578af0315a8fe7371730bd09611e9.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/2cea4943ebd4af597ad91da32df1e0fc9f9578af0315a8fe7371730bd09611e9.jpg)

![2cee0e24667b5d37696121b0a900b5f7fb712956953125021ecc105b8ebd83ad.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/2cee0e24667b5d37696121b0a900b5f7fb712956953125021ecc105b8ebd83ad.jpg)

![45255714f0e9767a090e80b8e75eee4985febf3d9768758a2126d7e405aef6f8.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/45255714f0e9767a090e80b8e75eee4985febf3d9768758a2126d7e405aef6f8.jpg)

![49356174d8814327518f49eccbfe3ae7d4f5e30469eb77dc2a9b539000c07772.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/49356174d8814327518f49eccbfe3ae7d4f5e30469eb77dc2a9b539000c07772.jpg)

![550846e82262c82d61bd6906f458aa068b7d8673de0dcbe62e6e866ad01c2afb.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/550846e82262c82d61bd6906f458aa068b7d8673de0dcbe62e6e866ad01c2afb.jpg)

![687e76128f4b19c72f79460eb6a0985ce86834f210fe18f13c54ae968ee7d0c0.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/687e76128f4b19c72f79460eb6a0985ce86834f210fe18f13c54ae968ee7d0c0.jpg)

![7712040ff393d0c19898752d1c35eb32a76a1ed2d4024155245a434849d3d394.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/7712040ff393d0c19898752d1c35eb32a76a1ed2d4024155245a434849d3d394.jpg)

![7bde33f47646fef608cef2b9fc693ac04489e4096944aeb2558db3b55d12bd9a.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/7bde33f47646fef608cef2b9fc693ac04489e4096944aeb2558db3b55d12bd9a.jpg)

![81f70d0a8ec7390ce254da0d46a865219d3d1b69a20b087f53b234ae92c5b6b0.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/81f70d0a8ec7390ce254da0d46a865219d3d1b69a20b087f53b234ae92c5b6b0.jpg)

![9310730de4cff48df8742f7455f107fd911098416fe70e26200a2d04204f0ae8.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/9310730de4cff48df8742f7455f107fd911098416fe70e26200a2d04204f0ae8.jpg)

![a0e17ca5f8cae343ae24af6bac19a91c5af6a735702f4cd7b5cdb47694402b4b.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/a0e17ca5f8cae343ae24af6bac19a91c5af6a735702f4cd7b5cdb47694402b4b.jpg)

![a4fac0334228c79d1f5fa96eb3e8553a40ad99dde27ade449e17bdfc31b4edcf.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/a4fac0334228c79d1f5fa96eb3e8553a40ad99dde27ade449e17bdfc31b4edcf.jpg)

![b019bc3aee4b84da0e259b6d5088ad8abc63b8f69311ad34de5356e99be49f32.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/b019bc3aee4b84da0e259b6d5088ad8abc63b8f69311ad34de5356e99be49f32.jpg)

![b5c9796d7bd91b617d0ea61e4b3ce50e6652ec8913907f9a4d5c05c72625b960.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/b5c9796d7bd91b617d0ea61e4b3ce50e6652ec8913907f9a4d5c05c72625b960.jpg)

![c01afc5bed8e44190ec601b3c9c49ccb9627ab3fb7a7461365dc8125582dfb12.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/c01afc5bed8e44190ec601b3c9c49ccb9627ab3fb7a7461365dc8125582dfb12.jpg)

![c5bea7586e8c40c4a72fa24761dede0b49cf762a0f50b574b7cd3741db4b51b2.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/c5bea7586e8c40c4a72fa24761dede0b49cf762a0f50b574b7cd3741db4b51b2.jpg)

![de68a829ceb2732a12c27473e7b9fbd52a49f8eb8b1945d6a1800452ea1d5c3d.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/de68a829ceb2732a12c27473e7b9fbd52a49f8eb8b1945d6a1800452ea1d5c3d.jpg)

![df7d0bcc8c550c1ac49e83cb4a02c69cd58997af0af5e032ede927fdf36ea0ed.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/df7d0bcc8c550c1ac49e83cb4a02c69cd58997af0af5e032ede927fdf36ea0ed.jpg)

![dfcde7eef658d9a1e0905ff92466bd89b5a201fc84bdc508780aa28b0ad41c15.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/dfcde7eef658d9a1e0905ff92466bd89b5a201fc84bdc508780aa28b0ad41c15.jpg)

![ea4f1c72216f1b69d6b8d5a90519c76a9b85d5fa62757200bf580f01ae120ce0.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/ea4f1c72216f1b69d6b8d5a90519c76a9b85d5fa62757200bf580f01ae120ce0.jpg)

![ecb915ab8a9624703b1c5f3458ba781b9381bdd8b299fb1d07c290ef3bcc3c28.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/ecb915ab8a9624703b1c5f3458ba781b9381bdd8b299fb1d07c290ef3bcc3c28.jpg)

![ef24ef12635c122b52d476ee98bfb30daa9672d00588e3b5c7b317a7a7e693d0.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/images/ef24ef12635c122b52d476ee98bfb30daa9672d00588e3b5c7b317a7a7e693d0.jpg)

### Tables

![0ae0b9ed5eebb3afa4967cceec25e7b6b215b41c89c3c244517a0b993cfbec5c.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/0ae0b9ed5eebb3afa4967cceec25e7b6b215b41c89c3c244517a0b993cfbec5c.jpg)

![1a4fc15201b5462a7faae0d1212068537d981c5614e209d60caa5cf9a405cf7c.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/1a4fc15201b5462a7faae0d1212068537d981c5614e209d60caa5cf9a405cf7c.jpg)

![1b8f552fe77e2100b1a4720b819ef03f6debb18e17c9718dfa9d5fb9220b8d1c.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/1b8f552fe77e2100b1a4720b819ef03f6debb18e17c9718dfa9d5fb9220b8d1c.jpg)

![2201a7ca94f1d831722adc8bd07cc36dde88a54186749b89180f9a3e98713ed2.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/2201a7ca94f1d831722adc8bd07cc36dde88a54186749b89180f9a3e98713ed2.jpg)

![346aaa279cc67ca2df0c9fce77a8d0340f8ab213403513c383235ffa627d8f71.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/346aaa279cc67ca2df0c9fce77a8d0340f8ab213403513c383235ffa627d8f71.jpg)

![51af1e529e4106b3479e75507de06ad7ead7425c772a4f42827cd7a5ec04354a.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/51af1e529e4106b3479e75507de06ad7ead7425c772a4f42827cd7a5ec04354a.jpg)

![701d79c1eb926ae6573b5097779f7240e1a4a784416dd65c46f37a0f2482aa93.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/701d79c1eb926ae6573b5097779f7240e1a4a784416dd65c46f37a0f2482aa93.jpg)

![7384473d4840051b9044278571b70f81a2f5f2d5f9370a3c0ae8bfa95607ed1a.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/7384473d4840051b9044278571b70f81a2f5f2d5f9370a3c0ae8bfa95607ed1a.jpg)

![785a05adc341f3b2fb5de505ea150047983e8c42ebfa70f8605519dcd034bfef.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/785a05adc341f3b2fb5de505ea150047983e8c42ebfa70f8605519dcd034bfef.jpg)

![7df1e03d7c9527cb34cf9e44bb2d400a31ea35bdad0ef20acbdc8ce0b2c8c9ed.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/7df1e03d7c9527cb34cf9e44bb2d400a31ea35bdad0ef20acbdc8ce0b2c8c9ed.jpg)

![827550015699b8a4ef01aab393506d3a93b83a47e01eb7c4c7c31e6031aec10d.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/827550015699b8a4ef01aab393506d3a93b83a47e01eb7c4c7c31e6031aec10d.jpg)

![87924e5d3c6453b772b28bf077363e59239f27c3bb34680f1c839ef9677dc04c.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/87924e5d3c6453b772b28bf077363e59239f27c3bb34680f1c839ef9677dc04c.jpg)

![909be086ea7d59f22b9608883459b9bbdf908e81f81c5ade111296298f113f51.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/909be086ea7d59f22b9608883459b9bbdf908e81f81c5ade111296298f113f51.jpg)

![abb53636a6911f11ba908100f2c4a1a5bea633d2321ecaed29ec0577d3cc5bcc.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/abb53636a6911f11ba908100f2c4a1a5bea633d2321ecaed29ec0577d3cc5bcc.jpg)

![c6345f1075078d1edc89cfc3e47c96ef08cb845ef1e913e5a77e01d156dfc884.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/c6345f1075078d1edc89cfc3e47c96ef08cb845ef1e913e5a77e01d156dfc884.jpg)

![ca9c406c1afd50c85766afb36477bff4e09812c7bd8180b1f340a940c795428e.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/ca9c406c1afd50c85766afb36477bff4e09812c7bd8180b1f340a940c795428e.jpg)

![d664b15a4fe49d89dd48729b5f300fa1fdb72bbd7b842673f433368a095ac5ab.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/d664b15a4fe49d89dd48729b5f300fa1fdb72bbd7b842673f433368a095ac5ab.jpg)

![d690ecb4581909a633dd4a35dda00fdddef552416373a30594398fae86b4f71c.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/d690ecb4581909a633dd4a35dda00fdddef552416373a30594398fae86b4f71c.jpg)

![dc517d4904f4e2bae1787e17c0abcb3c8139d9e6c9d3ed24c851416eccd78776.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/dc517d4904f4e2bae1787e17c0abcb3c8139d9e6c9d3ed24c851416eccd78776.jpg)

![e05ed6ba2a5c4c4b55be35424476313bc60c3c25f7b172f74adabb2c81b5a05c.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/e05ed6ba2a5c4c4b55be35424476313bc60c3c25f7b172f74adabb2c81b5a05c.jpg)

![e1c39453eccec79766313ec8cfc30d0e7424658957fccdd258dafd065eb6d9f9.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/e1c39453eccec79766313ec8cfc30d0e7424658957fccdd258dafd065eb6d9f9.jpg)

![ed0f4c94c2e51e40a5e5e65c3590e1332e737869cf14cccf86599463f1039cfd.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/ed0f4c94c2e51e40a5e5e65c3590e1332e737869cf14cccf86599463f1039cfd.jpg)

![f495e7ae4bf0105c8b06b2c8d54bdd730c7b7a910c51362493de38ee397931df.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/f495e7ae4bf0105c8b06b2c8d54bdd730c7b7a910c51362493de38ee397931df.jpg)

![fc45a83326d2de3798536fef73b7448c2770be8982ec21dc624cbf9dc8081141.jpg](../icml_results/56_CausalGame_%20Benchmarking%20Causal%20Thinking%20of%20LLM%20Agents%20in%20Games/tables/fc45a83326d2de3798536fef73b7448c2770be8982ec21dc624cbf9dc8081141.jpg)

<a id="paper-0056"></a>

## High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions


### Tables

![12a90bc6b619872814a2222b8345992f5521c9a8281d9132c3d497355a8be01d.jpg](../icml_results/57_High-Accuracy%20Sampling%20for%20Diffusion%20Models%20and%20Log-Concave%20Distributions/tables/12a90bc6b619872814a2222b8345992f5521c9a8281d9132c3d497355a8be01d.jpg)

<a id="paper-0057"></a>

## Hista and Numca: Estimate State Value Effectively for Large Language Model Reinforcement Learning


### Images

![04f5acf5ce2c68afd7baeac6ecba7e4b2b5df4d16d5482c3bb176ed2454c6c1a.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/04f5acf5ce2c68afd7baeac6ecba7e4b2b5df4d16d5482c3bb176ed2454c6c1a.jpg)

![0d0e55a7c22b16cd79a75a592bc10943dcc51ac9e4d7ca51f448c9addb6a5ad4.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/0d0e55a7c22b16cd79a75a592bc10943dcc51ac9e4d7ca51f448c9addb6a5ad4.jpg)

![1106bc9747c10c57611a7d73aba891368eb81b90773f121cf140f677846f8110.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/1106bc9747c10c57611a7d73aba891368eb81b90773f121cf140f677846f8110.jpg)

![2650e3e4c1507f82d8268f1930da75a8ca839af63e063731918efac72db2dbfd.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/2650e3e4c1507f82d8268f1930da75a8ca839af63e063731918efac72db2dbfd.jpg)

![32a0d7209617ac53a13e1bf636b7db4ab70021e85bf50e9b8514b6ae47698ea2.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/32a0d7209617ac53a13e1bf636b7db4ab70021e85bf50e9b8514b6ae47698ea2.jpg)

![88241cf1e9022e522248f8127aa91e50c5586efcc733b97d47fac1142b56e2ba.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/88241cf1e9022e522248f8127aa91e50c5586efcc733b97d47fac1142b56e2ba.jpg)

![8a85adde55aa49c1566359dfc06754cd3021e8d6d0ce225ac497c8408e71a4d9.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/8a85adde55aa49c1566359dfc06754cd3021e8d6d0ce225ac497c8408e71a4d9.jpg)

![8c38138ab5dcf860a512c6e749eb8b7252c019e5f511183e89919ad4cebd247a.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/8c38138ab5dcf860a512c6e749eb8b7252c019e5f511183e89919ad4cebd247a.jpg)

![e1433274dbae3b15ab4568d957b1439849e7c2b6997c86eb3f8517d2071efa87.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/images/e1433274dbae3b15ab4568d957b1439849e7c2b6997c86eb3f8517d2071efa87.jpg)

### Tables

![09f250276c73e80f334070281a738b56108c7c92ffa60b4115886ddac8137941.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/09f250276c73e80f334070281a738b56108c7c92ffa60b4115886ddac8137941.jpg)

![12881725ecce57a0aada27f775b16398ff0a94e002d86e7f55d4cb689192d577.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/12881725ecce57a0aada27f775b16398ff0a94e002d86e7f55d4cb689192d577.jpg)

![2bc29891b6765bdc11038872cb6debc27f98a53e263f15b1b00610694872ac60.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/2bc29891b6765bdc11038872cb6debc27f98a53e263f15b1b00610694872ac60.jpg)

![328de37f85af02efb18b8193937213487a381220651bd2d53ab5fe3a382a33c2.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/328de37f85af02efb18b8193937213487a381220651bd2d53ab5fe3a382a33c2.jpg)

![501ca199280bbf8796d4b6b20a12397479e8de6109eda5484f66783150571638.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/501ca199280bbf8796d4b6b20a12397479e8de6109eda5484f66783150571638.jpg)

![593857643598bdb5224ad23fbc01abd7a92b561c95fc6851b6e2f7cc9d097a8b.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/593857643598bdb5224ad23fbc01abd7a92b561c95fc6851b6e2f7cc9d097a8b.jpg)

![5ccab8305b31d12400eb8973414da26388cd85c4bf1aba100e3aeb0c76dc05e4.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/5ccab8305b31d12400eb8973414da26388cd85c4bf1aba100e3aeb0c76dc05e4.jpg)

![6c3a219c1c22490440378ff3dbe88a9d2576278e86c3cc34d53db625580ba5c4.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/6c3a219c1c22490440378ff3dbe88a9d2576278e86c3cc34d53db625580ba5c4.jpg)

![908d64e51f49f98673930381f809d99b79e830afe55e727e364ca2cdc2235bab.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/908d64e51f49f98673930381f809d99b79e830afe55e727e364ca2cdc2235bab.jpg)

![95a5575a5e6dc97f77e17c2463aec935e2b2229fd24d3c6db2f11727528aacbf.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/95a5575a5e6dc97f77e17c2463aec935e2b2229fd24d3c6db2f11727528aacbf.jpg)

![aba54a4cfb00899d8fa590137e37575fddf16e32aa0f451503c701a585dd0ec7.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/aba54a4cfb00899d8fa590137e37575fddf16e32aa0f451503c701a585dd0ec7.jpg)

![b2adea42f29aecc8dcbfba8f1fe2eed1c351833db3e5666b3510a643b7944a55.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/b2adea42f29aecc8dcbfba8f1fe2eed1c351833db3e5666b3510a643b7944a55.jpg)

![c376c72bedb9879af1dbb6333829a03680e7c13e5d4abcfac99d3e2721da8299.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/c376c72bedb9879af1dbb6333829a03680e7c13e5d4abcfac99d3e2721da8299.jpg)

![cd74473156d225ed74c88500e44be974c6c9bd59247399c65f2b26706d9691f0.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/cd74473156d225ed74c88500e44be974c6c9bd59247399c65f2b26706d9691f0.jpg)

![d052efaea5ae811b0b285bbcd7b77cf8cae38906a4469a2e6cbddf7d0b6d18fb.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/d052efaea5ae811b0b285bbcd7b77cf8cae38906a4469a2e6cbddf7d0b6d18fb.jpg)

![d36b41c3f9abdb68ac60e3e96ec45117b0da7001ad1a1b3de1f249616da6b3b4.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/d36b41c3f9abdb68ac60e3e96ec45117b0da7001ad1a1b3de1f249616da6b3b4.jpg)

![fa01b47e0eefdd6c953c1b657056364d7267b5f9632e4c587f93f5b2db47b742.jpg](../icml_results/58_Hista%20and%20Numca_%20Estimate%20State%20Value%20Effectively%20for%20Large%20Language%20Model%20Reinforcement%20Learning/tables/fa01b47e0eefdd6c953c1b657056364d7267b5f9632e4c587f93f5b2db47b742.jpg)

<a id="paper-0058"></a>

## Information-Theoretic Disentangled Latent Modeling with Conditional Diffusion for Incomplete Multi-View Clustering


### Images

![0febcc56249e22dac2bae2f86718b50ee08d983ba3aed5f619d70628f615dc04.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/0febcc56249e22dac2bae2f86718b50ee08d983ba3aed5f619d70628f615dc04.jpg)

![166654dac8547714e2d5c19517a6f4ae08be6e08a886bb5b38cb61c207f10d8d.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/166654dac8547714e2d5c19517a6f4ae08be6e08a886bb5b38cb61c207f10d8d.jpg)

![16901046f148d2349058f7e1f0b0821eb0ea497305d36e78add757a1a796dde2.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/16901046f148d2349058f7e1f0b0821eb0ea497305d36e78add757a1a796dde2.jpg)

![19c47c521b50ea440cf55835a623674692bb4e72a2c6358465ca9e57cf414adb.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/19c47c521b50ea440cf55835a623674692bb4e72a2c6358465ca9e57cf414adb.jpg)

![37dae69da6391b1cd62133f90ac4c560a354301a2126d242f578ee35ba2ba82e.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/37dae69da6391b1cd62133f90ac4c560a354301a2126d242f578ee35ba2ba82e.jpg)

![63b7062bccb20ad644d231b47febdb75768e1679c8a1697af2f35bb9624d65e4.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/63b7062bccb20ad644d231b47febdb75768e1679c8a1697af2f35bb9624d65e4.jpg)

![886f08c4337a61c0f5c1b75a666e6ec8393597af3a84d49b26b74237fa56a384.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/886f08c4337a61c0f5c1b75a666e6ec8393597af3a84d49b26b74237fa56a384.jpg)

![d0497fb940729443efceca1481be971f9ff27d4747c61f59e7a9540160800f91.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/images/d0497fb940729443efceca1481be971f9ff27d4747c61f59e7a9540160800f91.jpg)

### Tables

![0bccbf78637733580b1c2e5e9b67a3495551205144a1c046c1ff966227f1b95e.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/0bccbf78637733580b1c2e5e9b67a3495551205144a1c046c1ff966227f1b95e.jpg)

![0d9cbd479508d711c80dda3806f7a5538919c94df95ce273a5a3e4d17decd669.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/0d9cbd479508d711c80dda3806f7a5538919c94df95ce273a5a3e4d17decd669.jpg)

![10db392399357441381406d98904a3eaa2ebd56933fd38441bc9c3dbbef9d175.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/10db392399357441381406d98904a3eaa2ebd56933fd38441bc9c3dbbef9d175.jpg)

![27c81f49b067d4a455bdd1ab33fede3f13eb881d88d12d5402ab4f4ff91aff4d.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/27c81f49b067d4a455bdd1ab33fede3f13eb881d88d12d5402ab4f4ff91aff4d.jpg)

![43f6e5c815671c35c9f50eddb9b72bcdb5a97ff343e81aa7b708ad2603a7318c.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/43f6e5c815671c35c9f50eddb9b72bcdb5a97ff343e81aa7b708ad2603a7318c.jpg)

![644da7f3ff0cdbe1c9eed0703b469204d2bd03ed767f665171a9d82759cef9bf.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/644da7f3ff0cdbe1c9eed0703b469204d2bd03ed767f665171a9d82759cef9bf.jpg)

![c91170b37d20415c028e4bce3d70d4f22820e4ed6cbd5eb2399438541206d804.jpg](../icml_results/59_Information-Theoretic%20Disentangled%20Latent%20Modeling%20with%20Conditional%20Diffusion%20for%20Incomplete%20Multi-V/tables/c91170b37d20415c028e4bce3d70d4f22820e4ed6cbd5eb2399438541206d804.jpg)

<a id="paper-0059"></a>

## Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in Large Language Models


### Images

![3256adc905b52518fe11d20a3d7854177fe53eaee626566426daa432783a7eb0.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/3256adc905b52518fe11d20a3d7854177fe53eaee626566426daa432783a7eb0.jpg)

![3d5bbf920f4982fe96c3560f4c3ca8bfc110031a4b7514141d32657f8fc44693.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/3d5bbf920f4982fe96c3560f4c3ca8bfc110031a4b7514141d32657f8fc44693.jpg)

![43a3ddbdd869dae2a43644932b8ed8019fe4f86d07036fd9266b7d43ca282956.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/43a3ddbdd869dae2a43644932b8ed8019fe4f86d07036fd9266b7d43ca282956.jpg)

![51349f722093d0f9de808b7f893241acd25b56bd31ec4f8ac6750209858fc7f3.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/51349f722093d0f9de808b7f893241acd25b56bd31ec4f8ac6750209858fc7f3.jpg)

![876dbd458c13ea3702c1d403ca826eab7b35f696f7e0f4a27c2aeddf220395c9.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/876dbd458c13ea3702c1d403ca826eab7b35f696f7e0f4a27c2aeddf220395c9.jpg)

![8afef868676a33c93c0a612553c846231af65254baec1cd2fed702fd29763e15.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/8afef868676a33c93c0a612553c846231af65254baec1cd2fed702fd29763e15.jpg)

![a668696c33e3ef72d1ffd4d9e4dd0fb43e88b68d53703d551a9e66fadeb2d471.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/a668696c33e3ef72d1ffd4d9e4dd0fb43e88b68d53703d551a9e66fadeb2d471.jpg)

![b69dcb825fc2545dc7cdfb1d44d8903d978663e116c6958b7abce3edc3252415.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/images/b69dcb825fc2545dc7cdfb1d44d8903d978663e116c6958b7abce3edc3252415.jpg)

### Tables

![077132ca8defffe3ab347995b271e93120e6779ba69b009ea6209e1a9385ab02.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/077132ca8defffe3ab347995b271e93120e6779ba69b009ea6209e1a9385ab02.jpg)

![118f4b5cb056be43e69c825b5db194ff1b5a1bf92549066100e1f471e8ec4aa4.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/118f4b5cb056be43e69c825b5db194ff1b5a1bf92549066100e1f471e8ec4aa4.jpg)

![52a5f8f166ae6e427a62f014bf298e97c0a600738fe76ceb036eca61bd4ba30f.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/52a5f8f166ae6e427a62f014bf298e97c0a600738fe76ceb036eca61bd4ba30f.jpg)

![76c2770b32f636425a06de3505ce50301bf6caeed1bc3edf38d5586b1a4f1f26.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/76c2770b32f636425a06de3505ce50301bf6caeed1bc3edf38d5586b1a4f1f26.jpg)

![ab6283af490b08655629580a70199657f0e1997ba8c0096add98ace948f0ad01.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/ab6283af490b08655629580a70199657f0e1997ba8c0096add98ace948f0ad01.jpg)

![cf4cec3ffe56db00760d0fad64e90c759ef565e651f6d4e0cc5ce155cd491601.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/cf4cec3ffe56db00760d0fad64e90c759ef565e651f6d4e0cc5ce155cd491601.jpg)

![dd657b48c08e363ca78f889a9ec86fe7b7ac023edab6671da680d5fd09d6a830.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/dd657b48c08e363ca78f889a9ec86fe7b7ac023edab6671da680d5fd09d6a830.jpg)

![e3f6fdbe310f2252e840d9834a72a838a979a19e1c824946c594ad20acdd4928.jpg](../icml_results/60_Know%20More%2C%20Know%20Clearer_%20A%20Meta-Cognitive%20Framework%20for%20Knowledge%20Augmentation%20in%20Large%20Language%20Mod/tables/e3f6fdbe310f2252e840d9834a72a838a979a19e1c824946c594ad20acdd4928.jpg)

<a id="paper-0060"></a>

## Reward-free Alignment for Conflicting Objectives


### Images

![5c4a4fb94c09f3221bddc7445e316caa2398931672ccf835f53a1789de62e954.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/5c4a4fb94c09f3221bddc7445e316caa2398931672ccf835f53a1789de62e954.jpg)

![7405f562d44ed2723acdb1348737fa58bb374c2917b05900ab7026701c7febc6.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/7405f562d44ed2723acdb1348737fa58bb374c2917b05900ab7026701c7febc6.jpg)

![90c09c8f4f2254a631a4c6270ff83897209e8601caa6ab527a30179901e8c9cf.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/90c09c8f4f2254a631a4c6270ff83897209e8601caa6ab527a30179901e8c9cf.jpg)

![c186c581f2971db41e7262e9426a3515ff95a415f5a3d368c5277b0f74452ead.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/c186c581f2971db41e7262e9426a3515ff95a415f5a3d368c5277b0f74452ead.jpg)

![c4619374eb75a6b1189fc5ba3c11282371e7a69b562f91b41b8f4637e85475fb.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/c4619374eb75a6b1189fc5ba3c11282371e7a69b562f91b41b8f4637e85475fb.jpg)

![e429124b2c53d766a0b9ae4e66271419ae9be20377f2f0eb2d8d7a4aa86901ee.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/e429124b2c53d766a0b9ae4e66271419ae9be20377f2f0eb2d8d7a4aa86901ee.jpg)

![e5a52085ed230bc7268bd06e5e4902cba3703c27d89ee430064ebdd5995dd378.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/images/e5a52085ed230bc7268bd06e5e4902cba3703c27d89ee430064ebdd5995dd378.jpg)

### Tables

![060828b1b84c2f90bf73b2e069ba0749f8b6cd4c60f8763b2b52fd421c300299.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/tables/060828b1b84c2f90bf73b2e069ba0749f8b6cd4c60f8763b2b52fd421c300299.jpg)

![67758dc97cc2d4f6f5bb97d4937ed09460d5597df64c3408a40d02e130b7f0b9.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/tables/67758dc97cc2d4f6f5bb97d4937ed09460d5597df64c3408a40d02e130b7f0b9.jpg)

![9f04549a392088bae08123f88e966afa3c355f9aa0a352fdbaabddc3a512c8b2.jpg](../icml_results/61_Reward-free%20Alignment%20for%20Conflicting%20Objectives/tables/9f04549a392088bae08123f88e966afa3c355f9aa0a352fdbaabddc3a512c8b2.jpg)

<a id="paper-0061"></a>

## Focus and Dilution: The Multi-stage Learning Process of Attention


### Images

![5e65ca8c94621e518bde7e822b5971f9198022ebe262dc193c46a813eaee62e6.jpg](../icml_results/62_Focus%20and%20Dilution_%20The%20Multi-stage%20Learning%20Process%20of%20Attention/images/5e65ca8c94621e518bde7e822b5971f9198022ebe262dc193c46a813eaee62e6.jpg)

![5e82ef585e2252aac76c7d13dfaccd88ff8d0f27ce2eadfebbbd00f5c6bee5e0.jpg](../icml_results/62_Focus%20and%20Dilution_%20The%20Multi-stage%20Learning%20Process%20of%20Attention/images/5e82ef585e2252aac76c7d13dfaccd88ff8d0f27ce2eadfebbbd00f5c6bee5e0.jpg)

![688b821c6fed023fda220e438f5becc94c54d7068e95e3aca41768e060ccbcfa.jpg](../icml_results/62_Focus%20and%20Dilution_%20The%20Multi-stage%20Learning%20Process%20of%20Attention/images/688b821c6fed023fda220e438f5becc94c54d7068e95e3aca41768e060ccbcfa.jpg)

![9ec8a8e3945c6c7e6b42a6e682e3eb5eb2c87762b2b79e1784cef443d62299e2.jpg](../icml_results/62_Focus%20and%20Dilution_%20The%20Multi-stage%20Learning%20Process%20of%20Attention/images/9ec8a8e3945c6c7e6b42a6e682e3eb5eb2c87762b2b79e1784cef443d62299e2.jpg)

![a668fede06234eaf0da9c6df11cbfcd0eaad2ed692acfc75a9ce6285fc5607ec.jpg](../icml_results/62_Focus%20and%20Dilution_%20The%20Multi-stage%20Learning%20Process%20of%20Attention/images/a668fede06234eaf0da9c6df11cbfcd0eaad2ed692acfc75a9ce6285fc5607ec.jpg)

<a id="paper-0062"></a>

## HypoSpace: A Diagnostic Benchmark for Set-Valued Hypothesis Generation under Underdetermination and Sublinear Coverage Bounds


### Images

![2ba856cbdd1489dfb61eb0a1bb7f14c3bfa572f2ebd5e6d90663d7ef03e7ebba.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/images/2ba856cbdd1489dfb61eb0a1bb7f14c3bfa572f2ebd5e6d90663d7ef03e7ebba.jpg)

![587d71aa1a97806b84fd8832c0aed4a9369659811e955f3287728c7537b8eb5b.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/images/587d71aa1a97806b84fd8832c0aed4a9369659811e955f3287728c7537b8eb5b.jpg)

![63cccb661d90b5eec7561553049cfc4f6df4214bf9c38d1b02aa88bd110d1347.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/images/63cccb661d90b5eec7561553049cfc4f6df4214bf9c38d1b02aa88bd110d1347.jpg)

![8dc1e32a1b37e5585fc80983aa2b5a90c9e15aae92a59b964dd4ae0753d411b1.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/images/8dc1e32a1b37e5585fc80983aa2b5a90c9e15aae92a59b964dd4ae0753d411b1.jpg)

![e5a264e687059cc460bfba002e6495dc676804491b1e3f32a50a17852037dbd3.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/images/e5a264e687059cc460bfba002e6495dc676804491b1e3f32a50a17852037dbd3.jpg)

### Tables

![399410078e2706b078e8dc30e24f0e78dca76818f094c3d634c6de9d513fcbc9.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/399410078e2706b078e8dc30e24f0e78dca76818f094c3d634c6de9d513fcbc9.jpg)

![9318651d09d55431569fb4e329a359f7609140bc1d47f547376a7798ebc5b70d.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/9318651d09d55431569fb4e329a359f7609140bc1d47f547376a7798ebc5b70d.jpg)

![962cc0f0d429291f5ac9b893909aa549072ad8b5603443d32a7f66ed35bf2fc7.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/962cc0f0d429291f5ac9b893909aa549072ad8b5603443d32a7f66ed35bf2fc7.jpg)

![a2abcbcd61b60fb9ff06249d6624931c0e2356e68e3ef2c111a1792c622df40c.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/a2abcbcd61b60fb9ff06249d6624931c0e2356e68e3ef2c111a1792c622df40c.jpg)

![a472e0de2f6f2d8efb7edd964ad6f3e434023ca168d6bb4a8d625fe2e9bc8891.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/a472e0de2f6f2d8efb7edd964ad6f3e434023ca168d6bb4a8d625fe2e9bc8891.jpg)

![a9a0c9659cb107dbf9e92f9b4f23f9b440836a56e0ac2e3e0acc4e6fc7172990.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/a9a0c9659cb107dbf9e92f9b4f23f9b440836a56e0ac2e3e0acc4e6fc7172990.jpg)

![aa50c3419b1d4a50ba5f1cb726e3ddfc949f7fdbc56b7364ba8fb935d495afee.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/aa50c3419b1d4a50ba5f1cb726e3ddfc949f7fdbc56b7364ba8fb935d495afee.jpg)

![d1fb3bb5636998e97cf89f53497780f116bb43ad98764de4caa541afa530ba6c.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/d1fb3bb5636998e97cf89f53497780f116bb43ad98764de4caa541afa530ba6c.jpg)

![e0488d4eda3dda73fd04f9e778085c091dc63d6d8c4d12fe150882c0734dd97e.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/e0488d4eda3dda73fd04f9e778085c091dc63d6d8c4d12fe150882c0734dd97e.jpg)

![e0889792706d935656874eb4c20a765df170251d9f38fd0fb3ff0854704950f4.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/e0889792706d935656874eb4c20a765df170251d9f38fd0fb3ff0854704950f4.jpg)

![ebb0b7c2d86acd801da488380b854eb45b5552444e55a38b7384711b9898cc2b.jpg](../icml_results/63_HypoSpace_%20A%20Diagnostic%20Benchmark%20for%20Set-Valued%20Hypothesis%20Generation%20under%20Underdetermination%20and%20/tables/ebb0b7c2d86acd801da488380b854eb45b5552444e55a38b7384711b9898cc2b.jpg)

<a id="paper-0063"></a>

## Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement


### Images

![5a9207d4d86fec24d64ef0f47ddd2ef9af1e18653480a0f53a1f320bc08748b1.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/images/5a9207d4d86fec24d64ef0f47ddd2ef9af1e18653480a0f53a1f320bc08748b1.jpg)

![5bee251d86cb0e40ffbb409352ff923ee1f2fa08da2a6c0de73f0ba2691a427e.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/images/5bee251d86cb0e40ffbb409352ff923ee1f2fa08da2a6c0de73f0ba2691a427e.jpg)

![5f758e3aab38da037f336f6c299c324a323e3ae59d56db50cb5588bc377c5515.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/images/5f758e3aab38da037f336f6c299c324a323e3ae59d56db50cb5588bc377c5515.jpg)

![788231f9b8953d073906a8690a449e516cce4fc5c1e85fd8041ba1978af14e89.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/images/788231f9b8953d073906a8690a449e516cce4fc5c1e85fd8041ba1978af14e89.jpg)

![89c6be520f0ae07245d2bfd05fa6fa32ce889d63cfb3a4eb4eb0cb93a15dcb33.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/images/89c6be520f0ae07245d2bfd05fa6fa32ce889d63cfb3a4eb4eb0cb93a15dcb33.jpg)

![9eb47e364a1e1650d6a56b6c89eea14c45cfd6d5adf0edbc45550976e1fce5ec.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/images/9eb47e364a1e1650d6a56b6c89eea14c45cfd6d5adf0edbc45550976e1fce5ec.jpg)

### Tables

![07d160fe2353b26656a5b524284b3bcf0981ea3244d335311052620b453284c7.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/tables/07d160fe2353b26656a5b524284b3bcf0981ea3244d335311052620b453284c7.jpg)

![7f287e8fec999e094c35439c92a85aab9082d012c5f4338fff242ef23ba0ec3e.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/tables/7f287e8fec999e094c35439c92a85aab9082d012c5f4338fff242ef23ba0ec3e.jpg)

![83ea7071722a2661a528d7df24c3fb716a5ed00f821f81b916f78a877679dd6c.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/tables/83ea7071722a2661a528d7df24c3fb716a5ed00f821f81b916f78a877679dd6c.jpg)

![b574f542867a45460ec0decdc155b06a5b745ca534a47551d999cd43725e5243.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/tables/b574f542867a45460ec0decdc155b06a5b745ca534a47551d999cd43725e5243.jpg)

![d96ca850c6f86fcc6727201517135deca2a99949d5963207483f1ac95d51299e.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/tables/d96ca850c6f86fcc6727201517135deca2a99949d5963207483f1ac95d51299e.jpg)

![f95d325aa4d26b135a99f4d06323e65e72b9323a638d0d10ce8648180e1ff61d.jpg](../icml_results/64_Linear%20Causal%20Representation%20Learning%20by%20Topological%20Ordering%2C%20Pruning%2C%20and%20Disentanglement/tables/f95d325aa4d26b135a99f4d06323e65e72b9323a638d0d10ce8648180e1ff61d.jpg)

<a id="paper-0064"></a>

## TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control


### Images

![071e7289b529a216fc5743b80605451cc32fe5586dac7c9fcb06ff1c1c7e6a1e.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/071e7289b529a216fc5743b80605451cc32fe5586dac7c9fcb06ff1c1c7e6a1e.jpg)

![2666a33f1cc4b25e317374a2eddce71be28ea8886b5469095371320592d899b4.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/2666a33f1cc4b25e317374a2eddce71be28ea8886b5469095371320592d899b4.jpg)

![2d040821610538536935dd55ba1c590017aa8ccc72ea8ebcfc1fc1eabdbafb9b.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/2d040821610538536935dd55ba1c590017aa8ccc72ea8ebcfc1fc1eabdbafb9b.jpg)

![68a49801c05aa83eb0440c180866e4b9257e2bb4c0cfb080f00667792a8623f7.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/68a49801c05aa83eb0440c180866e4b9257e2bb4c0cfb080f00667792a8623f7.jpg)

![c5502d40974b5c15645bb4a4d91570c243644bcdb246181e17365933ac5ad980.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/c5502d40974b5c15645bb4a4d91570c243644bcdb246181e17365933ac5ad980.jpg)

![d5ada402addf6af5ecafa6f1d216c5fb845df3398805de6aa1369e8819a989ed.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/d5ada402addf6af5ecafa6f1d216c5fb845df3398805de6aa1369e8819a989ed.jpg)

![d6d508faa91e7e46f5e7a69c823758c204faca8a29142f5ea7a6a4d560eba3e8.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/d6d508faa91e7e46f5e7a69c823758c204faca8a29142f5ea7a6a4d560eba3e8.jpg)

![dab7f971b90da88e0d2da69364a1c4472deca71462bf51ea646b69951dba8e02.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/dab7f971b90da88e0d2da69364a1c4472deca71462bf51ea646b69951dba8e02.jpg)

![e352152aa3921de2e7d06f34ed157cd6c53abf5d2d332c5ac6427ad8ad6754e1.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/e352152aa3921de2e7d06f34ed157cd6c53abf5d2d332c5ac6427ad8ad6754e1.jpg)

![ed3704eaa46a5e5bd980116ff9f7e080837008f8b4357f29b54167af136cc909.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/images/ed3704eaa46a5e5bd980116ff9f7e080837008f8b4357f29b54167af136cc909.jpg)

### Tables

![2717e14ddd4c85f3446fd4f8e07a83dcdd5ef9cd729229eda290a6174311415d.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/2717e14ddd4c85f3446fd4f8e07a83dcdd5ef9cd729229eda290a6174311415d.jpg)

![30ddf83105d77409a3eab90e01848bc3c7719508c98a4bb83078c8b0e78f3c23.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/30ddf83105d77409a3eab90e01848bc3c7719508c98a4bb83078c8b0e78f3c23.jpg)

![3418518363cc47fe6ebb94bef4c83faf615d645c1fc3414e5923a68716bdb028.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/3418518363cc47fe6ebb94bef4c83faf615d645c1fc3414e5923a68716bdb028.jpg)

![49706abb50830b91a5c43d3840f9cb4aa1bf06478605b09eeec4c5e07439ec55.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/49706abb50830b91a5c43d3840f9cb4aa1bf06478605b09eeec4c5e07439ec55.jpg)

![4b69af6ddad875267248b5c51a8ce8fe14709740937489a3829ad4ab13825719.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/4b69af6ddad875267248b5c51a8ce8fe14709740937489a3829ad4ab13825719.jpg)

![4c56d1006d797cc9d3ba9bf28d257e9771c727028211c8c282aa3f53f37e450f.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/4c56d1006d797cc9d3ba9bf28d257e9771c727028211c8c282aa3f53f37e450f.jpg)

![53b59713c1984af16d873aab17e8807e2f7f39b7a30071f1d08d53e2c8cf5578.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/53b59713c1984af16d873aab17e8807e2f7f39b7a30071f1d08d53e2c8cf5578.jpg)

![5fd05197feff4835d7b5ed4e00123f0dbc3f5d5bec81394c8f072c1db2a3ddad.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/5fd05197feff4835d7b5ed4e00123f0dbc3f5d5bec81394c8f072c1db2a3ddad.jpg)

![776f7b3832ef56f0b2708f7fee10d3a1fb2be7d2420f263408f6bccf48601b7d.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/776f7b3832ef56f0b2708f7fee10d3a1fb2be7d2420f263408f6bccf48601b7d.jpg)

![833ea6cb9bccc30945b7f9e0fc0e0cbc1d35776d99aeec7b97def411965fc189.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/833ea6cb9bccc30945b7f9e0fc0e0cbc1d35776d99aeec7b97def411965fc189.jpg)

![8510aadece68e45c8ba69d31b32f9252d284aa9becb2d2abe86014af0e3ada46.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/8510aadece68e45c8ba69d31b32f9252d284aa9becb2d2abe86014af0e3ada46.jpg)

![8fc6cacaf504ebe58966e03612fc22eccb7457f62b0460b11e6d2843cbf1ecfa.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/8fc6cacaf504ebe58966e03612fc22eccb7457f62b0460b11e6d2843cbf1ecfa.jpg)

![92e4febefb83ca7c17349bbb87dc267b5e6d2293f2bbbe53cffcc0d552d62fda.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/92e4febefb83ca7c17349bbb87dc267b5e6d2293f2bbbe53cffcc0d552d62fda.jpg)

![9edb4128a72c42f8b70d6608cb4c887d57db2b79604b6cee38bc436ebfcf9297.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/9edb4128a72c42f8b70d6608cb4c887d57db2b79604b6cee38bc436ebfcf9297.jpg)

![a7b6c444fae60feec58116e11941a4354669dee4590ce926390fd388e3a0cd25.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/a7b6c444fae60feec58116e11941a4354669dee4590ce926390fd388e3a0cd25.jpg)

![dca8c1659a523b09dcb0ba3cb352a0ed1fde68b58795a49b59173bea2f66116e.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/dca8c1659a523b09dcb0ba3cb352a0ed1fde68b58795a49b59173bea2f66116e.jpg)

![eb0b38a9506f94d0c9c038a6fdfe2fe47f9c1dfd48577984d6c9f67dc14471fd.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/eb0b38a9506f94d0c9c038a6fdfe2fe47f9c1dfd48577984d6c9f67dc14471fd.jpg)

![edbdad8798a14ca40de5470ff331b144b9f81e7a35cbe929b702f7d1ebdb06cb.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/edbdad8798a14ca40de5470ff331b144b9f81e7a35cbe929b702f7d1ebdb06cb.jpg)

![faa214ae59558d43bde7614f71529fbc9409552a448a33701bb74ddd949990ac.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/faa214ae59558d43bde7614f71529fbc9409552a448a33701bb74ddd949990ac.jpg)

![ffab70d5856ac79ee0e06be6eb49594eb835fb6f1cfe4050739faafcf10e831b.jpg](../icml_results/65_TetraJet-v2_%20Accurate%20NVFP4%20Training%20for%20Large%20Language%20Models%20with%20Oscillation%20Suppression%20and%20Outl/tables/ffab70d5856ac79ee0e06be6eb49594eb835fb6f1cfe4050739faafcf10e831b.jpg)

<a id="paper-0065"></a>

## Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units


### Images

![08c85d8666dedf3c62da1859e6ae68c06bca87a265accfc13cb9a5bd02019105.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/08c85d8666dedf3c62da1859e6ae68c06bca87a265accfc13cb9a5bd02019105.jpg)

![09eb9a49db0c27480f0a26db7e9f51b14e089f6bd25193b16a1bc8df7545db46.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/09eb9a49db0c27480f0a26db7e9f51b14e089f6bd25193b16a1bc8df7545db46.jpg)

![1b3ec063ef61a333e3c4a21dd248efa460dfe17fd53b2d0b86f2187aedbce703.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/1b3ec063ef61a333e3c4a21dd248efa460dfe17fd53b2d0b86f2187aedbce703.jpg)

![1bdb883a3a62a7d455a63e314e958dc73d7b94d608035efbc79376980cffa832.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/1bdb883a3a62a7d455a63e314e958dc73d7b94d608035efbc79376980cffa832.jpg)

![26af43fa9151f715e8b1f01f41fe4fda1faa4c651248e7e6ec78e0bfe511ff96.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/26af43fa9151f715e8b1f01f41fe4fda1faa4c651248e7e6ec78e0bfe511ff96.jpg)

![356558ea974bb086e0dbaaccb5491687f3d1d10dcf4d2be757fe8853698a2cfe.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/356558ea974bb086e0dbaaccb5491687f3d1d10dcf4d2be757fe8853698a2cfe.jpg)

![3d7e8b0b6cfb5349942a517e5e0c09140ca8bcee115585e169a40dcdde6ebb5f.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/3d7e8b0b6cfb5349942a517e5e0c09140ca8bcee115585e169a40dcdde6ebb5f.jpg)

![3fcb364f68558dacff3dab895bed6bbc3ea81153899307b75aa4dd359e2dfb8e.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/3fcb364f68558dacff3dab895bed6bbc3ea81153899307b75aa4dd359e2dfb8e.jpg)

![542a51a0e8b645774a0e40fe83e0a414a28e72ca9aa7e637b8d2da0974965756.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/542a51a0e8b645774a0e40fe83e0a414a28e72ca9aa7e637b8d2da0974965756.jpg)

![63968e04b5ecc14c4a300e3bf6b3cb97f39318dd8b26ce35010c727457b2c48e.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/63968e04b5ecc14c4a300e3bf6b3cb97f39318dd8b26ce35010c727457b2c48e.jpg)

![6b9e2afad81d633d8f3d470cd2a4fa16d1d3e332a9ff984a5ffc7222922a3eb8.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/6b9e2afad81d633d8f3d470cd2a4fa16d1d3e332a9ff984a5ffc7222922a3eb8.jpg)

![b80f6ab6b8e0b8fa1478a56597cfb9fca62fd49d7d0a2ad87acc5fabf96c564a.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/b80f6ab6b8e0b8fa1478a56597cfb9fca62fd49d7d0a2ad87acc5fabf96c564a.jpg)

![be119026dc41bc68563d045861d84a61bacd746d61ceadef008def2ac4727b4b.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/be119026dc41bc68563d045861d84a61bacd746d61ceadef008def2ac4727b4b.jpg)

![daf22316a83aaa94a940b9f91a6b4654be800cb30742cd2a84a60ae42883a73e.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/daf22316a83aaa94a940b9f91a6b4654be800cb30742cd2a84a60ae42883a73e.jpg)

![f6c2f97685b292b51de79691c94048fe07de1ffa77bb74c4cfc7758583c0ce68.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/images/f6c2f97685b292b51de79691c94048fe07de1ffa77bb74c4cfc7758583c0ce68.jpg)

### Tables

![1d12d1182a194cf66b31ce315859cd2b3a8a8fff5c26e5ef8b4030274fa201ec.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/1d12d1182a194cf66b31ce315859cd2b3a8a8fff5c26e5ef8b4030274fa201ec.jpg)

![41a22ef92fe38a85aa5109400094a88e03128fcaadb52c191e69dc8bf1bc71b8.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/41a22ef92fe38a85aa5109400094a88e03128fcaadb52c191e69dc8bf1bc71b8.jpg)

![42db3cf6b7346a82b1ae36e5d58c186a6179eaf78224b8c8e5d54135588c04d3.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/42db3cf6b7346a82b1ae36e5d58c186a6179eaf78224b8c8e5d54135588c04d3.jpg)

![5b46990c484b8a1ed99d873f74d1de3a87b88fcd9c19ed95a064beaa26f26165.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/5b46990c484b8a1ed99d873f74d1de3a87b88fcd9c19ed95a064beaa26f26165.jpg)

![5b46b80a8266ab0584993137a6db765454579e92c07fb886843606e716ff88a2.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/5b46b80a8266ab0584993137a6db765454579e92c07fb886843606e716ff88a2.jpg)

![62a14c3a2138cfbd04df2a6a60c789cc439217a26608fc4009d76fa9c6acacdf.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/62a14c3a2138cfbd04df2a6a60c789cc439217a26608fc4009d76fa9c6acacdf.jpg)

![9c3828d6d2dcaf821e990120d6acfc424b93fb451b137088c071f4563518a30b.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/9c3828d6d2dcaf821e990120d6acfc424b93fb451b137088c071f4563518a30b.jpg)

![9f5b80a04d0c1ec7af3a5e011eada6659a978824f0384aa34cbe3243bc8c4568.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/9f5b80a04d0c1ec7af3a5e011eada6659a978824f0384aa34cbe3243bc8c4568.jpg)

![cb6618f462afb799105c18ae7f2c64cc4df6025f785f00378b6a3d3493c6ad64.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/cb6618f462afb799105c18ae7f2c64cc4df6025f785f00378b6a3d3493c6ad64.jpg)

![ff20a7cdf6b53b20b05df30551fd442da18eaba0b59a88a97e5995532cf34a20.jpg](../icml_results/66_Mechanistic%20Data%20Attribution_%20Tracing%20the%20Training%20Origins%20of%20Interpretable%20LLM%20Units/tables/ff20a7cdf6b53b20b05df30551fd442da18eaba0b59a88a97e5995532cf34a20.jpg)

<a id="paper-0066"></a>

## Beyond Global Alignment: Fine-Grained Motion-Language Retrieval via Pyramidal Shapley-Taylor Learning

### Images

![022de6fbc9130e7598be9293699c6e4bd954c10d31889408738508144b9d5063.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/022de6fbc9130e7598be9293699c6e4bd954c10d31889408738508144b9d5063.jpg)

![1add36664670ac49fa919b9945376685c61ff353c4a41aba2c25b4ff7de3228a.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/1add36664670ac49fa919b9945376685c61ff353c4a41aba2c25b4ff7de3228a.jpg)

![53f2d8d2ca34ccead80117788b73ce72d079a1e6dcc2d1eb27955e4ae4354b80.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/53f2d8d2ca34ccead80117788b73ce72d079a1e6dcc2d1eb27955e4ae4354b80.jpg)

![98615b7fadf48640352cbf8727a4e20d55b59569d17a71c80421160ba13774e3.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/98615b7fadf48640352cbf8727a4e20d55b59569d17a71c80421160ba13774e3.jpg)

![9af0e8ea9bc2ff63bb4b8c731e5d11ccd07bdbf5e04d1b208bd8782c2ea2a730.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/9af0e8ea9bc2ff63bb4b8c731e5d11ccd07bdbf5e04d1b208bd8782c2ea2a730.jpg)

![a770509983c82893a91fffd8c3f6106c1c70b85ba4d28cc02a3cb59dca333e54.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/a770509983c82893a91fffd8c3f6106c1c70b85ba4d28cc02a3cb59dca333e54.jpg)

![ae6817f79728acb0e9216ff40b06e6688a9e3a009fe7d331f1599e6bdfefef63.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/ae6817f79728acb0e9216ff40b06e6688a9e3a009fe7d331f1599e6bdfefef63.jpg)

![d1b08867050825379e8393608f63e5c85847f6200ea2d06e6b5cb2e788db8ea9.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/d1b08867050825379e8393608f63e5c85847f6200ea2d06e6b5cb2e788db8ea9.jpg)

![d8e52dda9905b750d17a204bfb51b5c44df474a7c4427ebf5ba5eb7ba582a024.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/d8e52dda9905b750d17a204bfb51b5c44df474a7c4427ebf5ba5eb7ba582a024.jpg)

![e89760e1a1733ed8c6ac33f39a1b6c2974f0ede2611e24429869884c282b4800.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/e89760e1a1733ed8c6ac33f39a1b6c2974f0ede2611e24429869884c282b4800.jpg)

![f81a543b6d7f17ecd5398962165a3a400b1acb78c581d2d338b6af49b27fc270.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/images/f81a543b6d7f17ecd5398962165a3a400b1acb78c581d2d338b6af49b27fc270.jpg)

### Tables

![0e4c80506b5a76adff0dc7c2a7fecfa474328147dc32a416b58774be17731377.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/0e4c80506b5a76adff0dc7c2a7fecfa474328147dc32a416b58774be17731377.jpg)

![27bb8ca4c9740ccc0c008a3bcc88432b860bba03f2713228b48829b50b4c2ea8.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/27bb8ca4c9740ccc0c008a3bcc88432b860bba03f2713228b48829b50b4c2ea8.jpg)

![2b5ec377343bae3d74ef4a306bf43eaeb2a23af9ffd2a8915cad4e9694d2da7b.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/2b5ec377343bae3d74ef4a306bf43eaeb2a23af9ffd2a8915cad4e9694d2da7b.jpg)

![39b629455cf0a31b9a86eb785c8a931952cc26862d3c8f97852ab1b98744b2bf.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/39b629455cf0a31b9a86eb785c8a931952cc26862d3c8f97852ab1b98744b2bf.jpg)

![60d661ff65d311d71529708465f8ec06ea6796f60837b8087c5cada4d66e5713.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/60d661ff65d311d71529708465f8ec06ea6796f60837b8087c5cada4d66e5713.jpg)

![64560f20f6e546bd88259cba023a7ac09d8bf26cec5499e4cabebe1db4deb8ef.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/64560f20f6e546bd88259cba023a7ac09d8bf26cec5499e4cabebe1db4deb8ef.jpg)

![66529fd12d33f57b48d0eea8aa1e3aa32e5cf3ee01bef3f36d6f26ad1042e86c.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/66529fd12d33f57b48d0eea8aa1e3aa32e5cf3ee01bef3f36d6f26ad1042e86c.jpg)

![97e04d732ebe63425163d9062f12590979c0db0c35947fb967490e6e08473613.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/97e04d732ebe63425163d9062f12590979c0db0c35947fb967490e6e08473613.jpg)

![a36e7efeef628cc514790df6dc0777fb50e8abc3e6983c5d74df576f0fefad43.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/a36e7efeef628cc514790df6dc0777fb50e8abc3e6983c5d74df576f0fefad43.jpg)

![f2a2e312c973dea73e72acb460748a2e0c25c06b2e8a344473b5ce4d25e8a91b.jpg](../icml_results/67_Beyond%20Global%20Alignment_%20Fine-Grained%20Motion-Language%20Retrieval%20via%20Pyramidal%20Shapley-Taylor%20Learnin/tables/f2a2e312c973dea73e72acb460748a2e0c25c06b2e8a344473b5ce4d25e8a91b.jpg)
