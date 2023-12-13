# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# 4M: Massively Multimodal Masked Modeling

## Authors
David Mizrahi, Roman Bachmann, Oğuzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, Amir Zamir

## Abstract
Current machine learning models for vision are often highly specialized andlimited to a single modality and task. In contrast, recent large languagemodels exhibit a wide range of capabilities, hinting at a possibility forsimilarly versatile models in computer vision. In this paper, we take a step inthis direction and propose a multimodal training scheme called 4M. It consistsof training a single unified Transformer encoder-decoder using a maskedmodeling objective across a wide range of input/output modalities - includingtext, images, geometric, and semantic modalities, as well as neural networkfeature maps. 4M achieves scalability by unifying the representation space ofall modalities through mapping them into discrete tokens and performingmultimodal masked modeling on a small randomized subset of tokens.  4M leads to models that exhibit several key capabilities: (1) they canperform a diverse set of vision tasks out of the box, (2) they excel whenfine-tuned for unseen downstream tasks or new input modalities, and (3) theycan function as a generative model that can be conditioned on arbitrarymodalities, enabling a wide variety of expressive multimodal editingcapabilities with remarkable flexibility.  Through experimental analyses, we demonstrate the potential of 4M fortraining versatile and scalable foundation models for vision tasks, setting thestage for further exploration in multimodal learning for vision and otherdomains.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior

## Authors
Fangfu Liu, Diankun Wu, Yi Wei, Yongming Rao, Yueqi Duan

## Abstract
Recently, 3D content creation from text prompts has demonstrated remarkableprogress by utilizing 2D and 3D diffusion models. While 3D diffusion modelsensure great multi-view consistency, their ability to generate high-quality anddiverse 3D assets is hindered by the limited 3D data. In contrast, 2D diffusionmodels find a distillation approach that achieves excellent generalization andrich details without any 3D data. However, 2D lifting methods suffer frominherent view-agnostic ambiguity thereby leading to serious multi-face Janusissues, where text prompts fail to provide sufficient guidance to learncoherent 3D results. Instead of retraining a costly viewpoint-aware model, westudy how to fully exploit easily accessible coarse 3D knowledge to enhance theprompts and guide 2D lifting optimization for refinement. In this paper, wepropose Sherpa3D, a new text-to-3D framework that achieves high-fidelity,generalizability, and geometric consistency simultaneously. Specifically, wedesign a pair of guiding strategies derived from the coarse 3D prior generatedby the 3D diffusion model: a structural guidance for geometric fidelity and asemantic guidance for 3D coherence. Employing the two types of guidance, the 2Ddiffusion model enriches the 3D content with diversified and high-qualityresults. Extensive experiments show the superiority of our Sherpa3D over thestate-of-the-art text-to-3D methods in terms of quality and 3D consistency.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior

## Authors
Fangfu Liu, Diankun Wu, Yi Wei, Yongming Rao, Yueqi Duan

## Abstract
Recently, 3D content creation from text prompts has demonstrated remarkableprogress by utilizing 2D and 3D diffusion models. While 3D diffusion modelsensure great multi-view consistency, their ability to generate high-quality anddiverse 3D assets is hindered by the limited 3D data. In contrast, 2D diffusionmodels find a distillation approach that achieves excellent generalization andrich details without any 3D data. However, 2D lifting methods suffer frominherent view-agnostic ambiguity thereby leading to serious multi-face Janusissues, where text prompts fail to provide sufficient guidance to learncoherent 3D results. Instead of retraining a costly viewpoint-aware model, westudy how to fully exploit easily accessible coarse 3D knowledge to enhance theprompts and guide 2D lifting optimization for refinement. In this paper, wepropose Sherpa3D, a new text-to-3D framework that achieves high-fidelity,generalizability, and geometric consistency simultaneously. Specifically, wedesign a pair of guiding strategies derived from the coarse 3D prior generatedby the 3D diffusion model: a structural guidance for geometric fidelity and asemantic guidance for 3D coherence. Employing the two types of guidance, the 2Ddiffusion model enriches the 3D content with diversified and high-qualityresults. Extensive experiments show the superiority of our Sherpa3D over thestate-of-the-art text-to-3D methods in terms of quality and 3D consistency.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior

## Authors
Fangfu Liu, Diankun Wu, Yi Wei, Yongming Rao, Yueqi Duan

## Abstract
Recently, 3D content creation from text prompts has demonstrated remarkableprogress by utilizing 2D and 3D diffusion models. While 3D diffusion modelsensure great multi-view consistency, their ability to generate high-quality anddiverse 3D assets is hindered by the limited 3D data. In contrast, 2D diffusionmodels find a distillation approach that achieves excellent generalization andrich details without any 3D data. However, 2D lifting methods suffer frominherent view-agnostic ambiguity thereby leading to serious multi-face Janusissues, where text prompts fail to provide sufficient guidance to learncoherent 3D results. Instead of retraining a costly viewpoint-aware model, westudy how to fully exploit easily accessible coarse 3D knowledge to enhance theprompts and guide 2D lifting optimization for refinement. In this paper, wepropose Sherpa3D, a new text-to-3D framework that achieves high-fidelity,generalizability, and geometric consistency simultaneously. Specifically, wedesign a pair of guiding strategies derived from the coarse 3D prior generatedby the 3D diffusion model: a structural guidance for geometric fidelity and asemantic guidance for 3D coherence. Employing the two types of guidance, the 2Ddiffusion model enriches the 3D content with diversified and high-qualityresults. Extensive experiments show the superiority of our Sherpa3D over thestate-of-the-art text-to-3D methods in terms of quality and 3D consistency.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Photorealistic Video Generation with Diffusion Models

## Authors
Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Li Fei-Fei, Irfan Essa, Lu Jiang, José Lezama

## Abstract
We present W.A.L.T, a transformer-based approach for photorealistic videogeneration via diffusion modeling. Our approach has two key design decisions.First, we use a causal encoder to jointly compress images and videos within aunified latent space, enabling training and generation across modalities.Second, for memory and training efficiency, we use a window attentionarchitecture tailored for joint spatial and spatiotemporal generative modeling.Taken together these design decisions enable us to achieve state-of-the-artperformance on established video (UCF-101 and Kinetics-600) and image(ImageNet) generation benchmarks without using classifier free guidance.Finally, we also train a cascade of three models for the task of text-to-videogeneration consisting of a base latent video diffusion model, and two videosuper-resolution diffusion models to generate videos of $512 \times 896$resolution at $8$ frames per second.

---

# Information theory for model reduction in stochastic dynamical systems

## Authors
Matthew S. Schmitt, Maciej Koch-Janusz, Michel Fruchart, Daniel S. Seara, Vincenzo Vitelli

## Abstract
Model reduction is the construction of simple yet predictive descriptions ofthe dynamics of many-body systems in terms of a few relevant variables. Aprerequisite to model reduction is the identification of these relevantvariables, a task for which no general method exists. Here, we develop asystematic approach based on the information bottleneck to identify therelevant variables, defined as those most predictive of the future. Weelucidate analytically the relation between these relevant variables and theeigenfunctions of the transfer operator describing the dynamics. Further, weshow that in the limit of high compression, the relevant variables are directlydetermined by the slowest-decaying eigenfunctions. Our information-basedapproach indicates when to optimally stop increasing the complexity of thereduced model. Further, it provides a firm foundation to constructinterpretable deep learning tools that perform model reduction. We illustratehow these tools work on benchmark dynamical systems and deploy them onuncurated datasets, such as satellite movies of atmospheric flows downloadeddirectly from YouTube.

---

# Mitigating Perspective Distortion-induced Shape Ambiguity in Image Crops

## Authors
Aditya Prakash, Arjun Gupta, Saurabh Gupta

## Abstract
Objects undergo varying amounts of perspective distortion as they move acrossa camera's field of view. Models for predicting 3D from a single image oftenwork with crops around the object of interest and ignore the location of theobject in the camera's field of view. We note that ignoring this locationinformation further exaggerates the inherent ambiguity in making 3D inferencesfrom 2D images and can prevent models from even fitting to the training data.To mitigate this ambiguity, we propose Intrinsics-Aware Positional Encoding(KPE), which incorporates information about the location of crops in the imageand camera intrinsics. Experiments on three popular 3D-from-a-single-imagebenchmarks: depth prediction on NYU, 3D object detection on KITTI & nuScenes,and predicting 3D shapes of articulated objects on ARCTIC, show the benefits ofKPE.

---

# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

## Authors
Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang

## Abstract
In machine learning, generalization against distribution shifts -- where deployment conditions diverge from the training scenarios -- is crucial, particularly in fields like climate modeling, biomedicine, and autonomous driving. The emergence of foundation models, distinguished by their extensive pretraining and task versatility, has led to an increased interest in their adaptability to distribution shifts. GPT-4V(ision) acts as the most advanced publicly accessible multimodal foundation model, with extensive applications across various domains, including anomaly detection, video understanding, image generation, and medical diagnosis. However, its robustness against data distributions remains largely underexplored. Addressing this gap, this study rigorously evaluates GPT-4V's adaptability and generalization capabilities in dynamic environments, benchmarking against prominent models like CLIP and LLaVA. We delve into GPT-4V's zero-shot generalization across 13 diverse datasets spanning natural, medical, and molecular domains. We further investigate its adaptability to controlled data perturbations and examine the efficacy of in-context learning as a tool to enhance its adaptation. Our findings delineate GPT-4V's capability boundaries in distribution shifts, shedding light on its strengths and limitations across various scenarios. Importantly, this investigation contributes to our understanding of how AI foundation models generalize to distribution shifts, offering pivotal insights into their adaptability and robustness. Code is publicly available at https://github.com/jameszhou-gl/gpt-4v-distribution-shift.

---

# MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception

## Authors
Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, Jing Shao

## Abstract
It is a long-lasting goal to design an embodied system that can solve long-horizon open-world tasks in human-like ways. However, existing approaches usually struggle with compound difficulties caused by the logic-aware decomposition and context-aware execution of these tasks. To this end, we introduce MP5, an open-ended multimodal embodied system built upon the challenging Minecraft simulator, which can decompose feasible sub-objectives, design sophisticated situation-aware plans, and perform embodied action control, with frequent communication with a goal-conditioned active perception scheme. Specifically, MP5 is developed on top of recent advances in Multimodal Large Language Models (MLLMs), and the system is modulated into functional modules that can be scheduled and collaborated to ultimately solve pre-defined context- and process-dependent tasks. Extensive experiments prove that MP5 can achieve a 22% success rate on difficult process-dependent tasks and a 91% success rate on tasks that heavily depend on the context. Moreover, MP5 exhibits a remarkable ability to address many open-ended tasks that are entirely novel.

---

# diff History for Long-Context Language Agents

## Authors
Ulyana Piterbarg, Lerrel Pinto, Rob Fergus

## Abstract
Language Models (LMs) offer an exciting solution for general-purpose embodied control. However, a key technical issue arises when using an LM-based controller: environment observations must be converted to text, which coupled with history, leads to prohibitively large textual prompts. As a result, prior work in LM agents is limited to restricted domains with either small observation size or minimal needs for interaction history. In this paper, we introduce a simple and highly effective solution to these issues. We exploit the fact that consecutive text observations have high similarity and propose to compress them via the Unix diff command. We demonstrate our approach in NetHack, a complex rogue-like video game, that requires long-horizon reasoning for decision-making and is far from solved, particularly for neural agents. Diff history offers an average of 4x increase in the length of the text-based interaction history available to the LM. This observational compression along with the benefits of abstraction yields a 7x improvement in game score on held-out environment instances over state-of-the-art baselines. It also outperforms prior agents that use visual observations by over 40%.

---

# PEEKABOO: Interactive Video Generation via Masked-Diffusion

## Authors
Yash Jain, Anshul Nasery, Vibhav Vineet, Harkirat Behl

## Abstract
Recently there has been a lot of progress in text-to-video generation, with state-of-the-art models being capable of generating high quality, realistic videos. However, these models lack the capability for users to interactively control and generate videos, which can potentially unlock new areas of application. As a first step towards this goal, we tackle the problem of endowing diffusion-based video generation models with interactive spatio-temporal control over their output. To this end, we take inspiration from the recent advances in segmentation literature to propose a novel spatio-temporal masked attention module - Peekaboo. This module is a training-free, no-inference-overhead addition to off-the-shelf video generation models which enables spatio-temporal control. We also propose an evaluation benchmark for the interactive video generation task. Through extensive qualitative and quantitative evaluation, we establish that Peekaboo enables control video generation and even obtains a gain of upto 3.8x in mIoU over baseline models.

---

# Cross-modal Contrastive Learning with Asymmetric Co-attention Network for Video Moment Retrieval

## Authors
Love Panta, Prashant Shrestha, Brabeem Sapkota, Amrita Bhattarai, Suresh Manandhar, Anand Kumar Sah

## Abstract
Video moment retrieval is a challenging task requiring fine-grained interactions between video and text modalities. Recent work in image-text pretraining has demonstrated that most existing pretrained models suffer from information asymmetry due to the difference in length between visual and textual sequences. We question whether the same problem also exists in the video-text domain with an auxiliary need to preserve both spatial and temporal information. Thus, we evaluate a recently proposed solution involving the addition of an asymmetric co-attention network for video grounding tasks. Additionally, we incorporate momentum contrastive loss for robust, discriminative representation learning in both modalities. We note that the integration of these supplementary modules yields better performance compared to state-of-the-art models on the TACoS dataset and comparable results on ActivityNet Captions, all while utilizing significantly fewer parameters with respect to baseline.

---

# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

## Authors
Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang

## Abstract
In machine learning, generalization against distribution shifts -- where deployment conditions diverge from the training scenarios -- is crucial, particularly in fields like climate modeling, biomedicine, and autonomous driving. The emergence of foundation models, distinguished by their extensive pretraining and task versatility, has led to an increased interest in their adaptability to distribution shifts. GPT-4V(ision) acts as the most advanced publicly accessible multimodal foundation model, with extensive applications across various domains, including anomaly detection, video understanding, image generation, and medical diagnosis. However, its robustness against data distributions remains largely underexplored. Addressing this gap, this study rigorously evaluates GPT-4V's adaptability and generalization capabilities in dynamic environments, benchmarking against prominent models like CLIP and LLaVA. We delve into GPT-4V's zero-shot generalization across 13 diverse datasets spanning natural, medical, and molecular domains. We further investigate its adaptability to controlled data perturbations and examine the efficacy of in-context learning as a tool to enhance its adaptation. Our findings delineate GPT-4V's capability boundaries in distribution shifts, shedding light on its strengths and limitations across various scenarios. Importantly, this investigation contributes to our understanding of how AI foundation models generalize to distribution shifts, offering pivotal insights into their adaptability and robustness. Code is publicly available at https://github.com/jameszhou-gl/gpt-4v-distribution-shift.

---

# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

## Authors
Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang

## Abstract
In machine learning, generalization against distribution shifts -- where deployment conditions diverge from the training scenarios -- is crucial, particularly in fields like climate modeling, biomedicine, and autonomous driving. The emergence of foundation models, distinguished by their extensive pretraining and task versatility, has led to an increased interest in their adaptability to distribution shifts. GPT-4V(ision) acts as the most advanced publicly accessible multimodal foundation model, with extensive applications across various domains, including anomaly detection, video understanding, image generation, and medical diagnosis. However, its robustness against data distributions remains largely underexplored. Addressing this gap, this study rigorously evaluates GPT-4V's adaptability and generalization capabilities in dynamic environments, benchmarking against prominent models like CLIP and LLaVA. We delve into GPT-4V's zero-shot generalization across 13 diverse datasets spanning natural, medical, and molecular domains. We further investigate its adaptability to controlled data perturbations and examine the efficacy of in-context learning as a tool to enhance its adaptation. Our findings delineate GPT-4V's capability boundaries in distribution shifts, shedding light on its strengths and limitations across various scenarios. Importantly, this investigation contributes to our understanding of how AI foundation models generalize to distribution shifts, offering pivotal insights into their adaptability and robustness. Code is publicly available at https://github.com/jameszhou-gl/gpt-4v-distribution-shift.

---

# How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

## Authors
Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Xing Xie, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang

## Abstract
In machine learning, generalization against distribution shifts -- where deployment conditions diverge from the training scenarios -- is crucial, particularly in fields like climate modeling, biomedicine, and autonomous driving. The emergence of foundation models, distinguished by their extensive pretraining and task versatility, has led to an increased interest in their adaptability to distribution shifts. GPT-4V(ision) acts as the most advanced publicly accessible multimodal foundation model, with extensive applications across various domains, including anomaly detection, video understanding, image generation, and medical diagnosis. However, its robustness against data distributions remains largely underexplored. Addressing this gap, this study rigorously evaluates GPT-4V's adaptability and generalization capabilities in dynamic environments, benchmarking against prominent models like CLIP and LLaVA. We delve into GPT-4V's zero-shot generalization across 13 diverse datasets spanning natural, medical, and molecular domains. We further investigate its adaptability to controlled data perturbations and examine the efficacy of in-context learning as a tool to enhance its adaptation. Our findings delineate GPT-4V's capability boundaries in distribution shifts, shedding light on its strengths and limitations across various scenarios. Importantly, this investigation contributes to our understanding of how AI foundation models generalize to distribution shifts, offering pivotal insights into their adaptability and robustness. Code is publicly available at https://github.com/jameszhou-gl/gpt-4v-distribution-shift.

---

