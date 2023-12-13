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

