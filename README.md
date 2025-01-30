# Detecting Conversational Mental Manipulation with Intent-Aware Prompting

## 1. Introduction

This repository is for the paper **'Detecting Conversational Mental Manipulation with Intent-Aware Prompting'**. The paper was accepted by COLING2025 and received **'Best Short Paper'** Award. The paper is available at this [link](https://arxiv.org/abs/2412.08414).

Here is the abstract of this paper. Mental manipulation severely undermines mental wellness by covertly and negatively distorting decision-making. While there is an increasing interest in mental health care within the natural language processing community, progress in tackling manipulation remains limited due to the complexity of detecting subtle, covert tactics in conversations. In this paper, we propose Intent-Aware Prompting (IAP), a novel approach for detecting mental manipulations using large language models (LLMs), providing a deeper understanding of manipulative tactics by capturing the underlying intents of participants. Experimental results on the MentalManip dataset demonstrate superior effectiveness of IAP against other advanced prompting strategies. Notably, our approach substantially reduces false negatives, helping detect more instances of mental manipulation with minimal misjudgment of positive cases.
![Framework](https://github.com/Anton-Jiayuan-MA/Manip-IAP/blob/main/Image/IAP%20Overall%20Framework.png)

## 2. File Structure of This Repository

```
Manip-IAP/
├── README.md
├── Code/  # contains code from dataset preparation to all prompt methods.
├── Dataset/  # contains the original MentalManip dataset and generated data during experiments.
├── Image/  # contains images used in the paper.
```

## 3. To Run The Experiments

### 3.1 Setup

|Packages|Version|
|-|-|
|pandas|2.1.4|
|numpy|1.26.4|
|sklearn|1.4.2|
|openai|1.51.0|

### 3.2 Run the Code

All users could run the code in this repository following the indices in the file names one by one.

## 4. Citation

```
@inproceedings{Ma2024DetectingCM,
  title={Detecting Conversational Mental Manipulation with Intent-Aware Prompting},
  author={Jiayuan Ma and Hongbin Na and Zimu Wang and Yining Hua and Yuegen Liu and Wei Wang and Ling Chen},
  booktitle={International Conference on Computational Linguistics},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:274638834}
}
```
