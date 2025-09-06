

If your ultimate goal is to learn about LLMs (Large Language Models), it's recommended to start with **PyTorch**, as most current LLM research and open-source implementations (like LLaMA, GPT, Mistral, etc.) are based on PyTorch[cite: 1].

Recommended Learning Path (From Scratch to LLM) [cite: 1]

✅ **Phase 1: Master Deep Learning Fundamentals** [cite: 1]
* Start with PyTorch (official tutorials + "Deep Learning from Scratch")[cite: 1].
* Learn the basic principles of neural networks (MLP, CNN, RNN)[cite: 1].
* Get familiar with the Transformer (the core architecture)[cite: 1].

✅ **Phase 2: Learn NLP-related Technologies** [cite: 1]
* Understand tokenization (BPE, WordPiece)[cite: 1].
* Study the Transformer and Self-Attention[cite: 1].
* Practice with small NLP tasks (text classification, named entity recognition, etc.)[cite: 1].

✅ **Phase 3: Dive Deep into LLM Training and Fine-tuning** [cite: 1]
* Study the architectures of GPT, BERT, and LLaMA[cite: 1].
* Learn LoRA and QLoRA (low-cost fine-tuning)[cite: 1].
* Try training or fine-tuning an LLM using Hugging Face Transformers[cite: 1].

✅ **Phase 4: Optimization and Deployment** [cite: 1]
* Understand Quantization and Distillation[cite: 1].
* Try deploying models with vLLM, TGI, and TensorRT-LLM[cite: 1].

---

## **🧠 1. Pre-training Tasks (Masked LM, Causal LM) & Fine-tuning Techniques** [cite: 2]

### ✅ **Basic Concepts** [cite: 2]

* **Masked LM (e.g., BERT)**: Randomly masks some tokens in the input and has the model predict them[cite: 2].
* **Causal LM (e.g., GPT)**: Predicts one token at a time by only looking at the left-side context, which is suitable for generation tasks[cite: 2].
* **Fine-tuning Techniques**: Such as full-parameter fine-tuning, Adapter, LoRA, QLoRA, PEFT, etc.[cite: 2].

### 📘 **Recommended Resources** [cite: 2]

#### 📗 **Beginner (Quickly Grasp Concepts)** [cite: 2]

* ***The Illustrated Transformer*** by Jay Alammar [cite: 2]
    👉 [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/) [cite: 2]

    > Explains the Transformer visually, perfect for beginners[cite: 2].

* **Hugging Face Course** (Recommended!) [cite: 2]
    👉 [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course) [cite: 2]
    * Lesson 3: Pretraining [cite: 2]
    * Lesson 4: Fine-tuning [cite: 2]

    > Designed for actual model training using the `transformers` library, highly practical[cite: 2].

* **Chinese:动手学Transformer** (Hands-on Transformer) [cite: 2]
    👉 GitHub: [https://github.com/datawhalechina/torch-transformers](https://github.com/datawhalechina/torch-transformers) [cite: 2]

    > Covers everything from the Transformer architecture to pre-training tasks and downstream fine-tuning[cite: 2].

#### 📘 **Advanced (In-depth Mechanisms and Practice)** [cite: 2]

* **Paper Recommendations**: [cite: 2]

    * BERT Original Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) [cite: 2]
    * GPT Original Paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [cite: 2]

* **Practical Recommendations (LoRA/PEFT)** [cite: 2]

    * PEFT Official Documentation: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index) [cite: 2]
    * LoRA [cite: 3] Tutorial Chinese Translation (Zhihu) [cite: 3]
    👉 [https://zhuanlan.zhihu.com/p/610674306](https://zhuanlan.zhihu.com/p/610674306) [cite: 3]

---

## **🎯 2. Prompt Engineering & In-Context Learning Principles** [cite: 3]

### ✅ **Core Concepts** [cite: 3]

* **Prompt Engineering**: Using a specific template to guide a model to generate more accurate output[cite: 3].
* **In-Context Learning (ICL)**: Without modifying parameters, you give the model some examples, and it learns how to complete the task[cite: 3].

### 📘 **Recommended Resources** [cite: 3]

#### 📗 **Beginner Understanding** [cite: 3]

* **Prompt Engineering Guide** (Very comprehensive) [cite: 3]
    👉 [https://github.com/dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) [cite: 3]

    > Covers common strategies like Few-shot, Chain-of-Thought (CoT), ReAct, and is continuously updated[cite: 3].

* **OpenAI Official Tutorial: Prompt Engineering for Developers** [cite: 3]
    👉 [https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering) [cite: 3]

    > Designed specifically for ChatGPT/GPT-4 users[cite: 3].

* **Chinese Explanation Videos (Bilibili recommended)** [cite: 3]

    * Search for keywords like "Prompt Engineering 原理" (Prompt Engineering Principles) or "In-Context Learning 机制" (In-Context Learning Mechanisms)[cite: 3]. There are many good NLP bloggers[cite: 3].
    * Recommended: Videos from PaddlePaddle and the Hugging Face Chinese community[cite: 3].

#### 📘 **Advanced Resources** [cite: 4]

* **Paper Recommendations**: [cite: 4]

    * GPT-3 Original Paper (Source of ICL theory) [cite: 4]
        👉 [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) [cite: 4]
    * Chain-of-Thought Prompting paper [cite: 4]
        👉 [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903) [cite: 4]

        > Significantly improves performance on complex reasoning problems[cite: 4].

* **Video Course: CS324 @ Stanford (Large [cite: 4] Language Models)** [cite: 4]
    👉 [https://web.stanford.edu/class/cs324/](https://web.stanford.edu/class/cs324/) [cite: 4]

    > Covers the latest cutting-edge topics, including ICL, prompt, alignment, and other core LLM subjects[cite: 4].

---

## 🧩 **Suggested Learning Order (Combining both parts)** [cite: 4]

1.  First, use Hugging Face to learn the Transformer architecture, pre-training, and fine-tuning mechanisms[cite: 4].
2.  Then, read the GPT-3 paper to understand what in-context learning is[cite: 4].
3.  Next, practice various prompt engineering techniques (few-shot, CoT, ReAct)[cite: 4].
4.  Finally, dive into efficient fine-tuning methods like LoRA/QLoRA[cite: 4].

---

This is great! Here is a customized **8-Week LLM Learning Plan** for you, with the goal of: mastering the core principles of LLMs, fine-tuning methods, Prompt Engineering, and inference mechanisms, ultimately being able to independently build and deploy an LLM application[cite: 4].

---

# 🧭 **8-Week LLM Learning Plan (Engineering + Research Focus)** [cite: 4]

### 👨‍💻 **Approximate Weekly Time Commitment: 8-12 hours (Flexible)** [cite: 4]

Each week includes three parts: [cite: 4]

* 🧠 Theory Learning [cite: 4]
* 🛠️ Hands-on Practice [cite: 4]
* 🔗 Resource Links [cite: 4]

---

## **Week 1: Transformer Architecture & Pre-training Concepts** [cite: 4]

### 🧠 **Theory** [cite: 4]

* Transformer Architecture and Attention [cite: 4]
* Masked LM vs. Causal LM [cite: 4]
* Difference between Pre-training and Downstream Tasks [cite: 4]

### 🛠️ **Practice** [cite: 4]

* Load BERT and GPT-2 with `transformers` [cite: 4]
* Use GPT-2 to generate text and experience causal LM [cite: 4]

### 🔗 **Resources** [cite: 4]

* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) [cite: 4]
* [Hugging Face NLP Course Chapter 1–4](https://huggingface.co/learn/nlp-course) [cite: 4]
* [Transformers Documentation Introduction](https://huggingface.co/docs/transformers/index) [cite: 4]

---

## **Week 2: Hugging [cite: 5] Face Practice + Fine-tuning BERT on Custom Data** [cite: 5]

### 🧠 **Theory** [cite: 5]

* How Tokenizers Work [cite: 5]
* Basic Fine-tuning Workflow [cite: 5]

### 🛠️ **Practice** [cite: 5]

* Fine-tune BERT for sentiment analysis or text classification (IMDb dataset) [cite: 5]
* Use Datasets + Trainer API [cite: 5]

### 🔗 **Resources** [cite: 5]

* [Hugging Face Docs - Fine-tuning BERT](https://huggingface.co/docs/transformers/training) [cite: 5]
* [IMDb Dataset](https://huggingface.co/datasets/imdb) [cite: 5]
* [Official Fine-tuning Tutorial Notebook](https://github.com/huggingface/notebooks/blob/main/course/en/chapter3/section3.ipynb) [cite: 5]

---

## **Week 3: Causal LM Training + GPT-2 Text Generation Techniques** [cite: 5]

### 🧠 **Theory** [cite: 5]

* Decoding Strategies (Greedy, Beam, Top-k, Top-p) [cite: 5]
* Meaning of Temperature and Repetition Penalty [cite: 5]

### 🛠️ **Practice** [cite: 5]

* Use GPT-2 for text generation tasks (e.g., writing poems, dialogues) [cite: 5]
* Compare the effects of different decoding strategies [cite: 5]

### 🔗 **Resources** [cite: 5]

* [Detailed Blog on Text Generation](https://huggingface.co/blog/how-to-generate) [cite: 5]
* [GPT2 Text Generation Notebook Example](https://github.com/huggingface/notebooks/blob/main/examples/text_generation.ipynb) [cite: 5]

---

## **Week 4: Prompt Engineering & In-Context Learning** [cite: 5]

### 🧠 **Theory** [cite: 5]

* Zero-shot / Few-shot Prompting [cite: 5]
* Chain-of-Thought (CoT) Prompting [cite: 5]
* Introduction to ReAct Models [cite: 5]

### 🛠️ **Practice** [cite: 5]

* Construct prompts to perform sentiment classification and solve logic problems [cite: 5]
* Compare performance with and without CoT [cite: 5]

### 🔗 **Resources** [cite: 5]

* [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) [cite: 5]
* [CoT Prompting Paper](https://arxiv.org/abs/2201.11903) [cite: 5]
* [OpenAI Prompt Guide](https://platform.openai.com/docs/guides/prompt-engineering) [cite: 5]

---

## **Week 5: LoRA / PEFT Fine-tuning Techniques** [cite: 5]

### 🧠 **Theory** [cite: 6]

* Full-parameter fine-tuning [cite: 6] vs. LoRA/Adapter [cite: 6]
* PEFT core idea: freezing the large model and only fine-tuning a few layers [cite: 6]

### 🛠️ **Practice** [cite: 6]

* Fine-tune LLaMA/Mistral/ChatGLM models using `peft` [cite: 6]
* Try using QLoRA to train models in a low-VRAM environment [cite: 6]

### 🔗 **Resources** [cite: 6]

* [PEFT Official Documentation](https://huggingface.co/docs/peft) [cite: 6]
* [QLoRA Original Paper](https://arxiv.org/abs/2305.14314) [cite: 6]
* [Chinese QLoRA Tutorial](https://zhuanlan.zhihu.com/p/640898922) [cite: 6]

---

## **Week 6: LLM Application Development + Gradio Frontend** [cite: 6]

### 🧠 **Theory** [cite: 6]

* How to deploy a fine-tuned model [cite: 6]
* Gradio/Streamlit frontend frameworks [cite: 6]

### 🛠️ **Practice** [cite: 6]

* Build a Chatbot interface [cite: 6]
* Support prompt input, setting temperature, max tokens, and other parameters [cite: 6]

### 🔗 **Resources** [cite: 6]

* [Gradio Getting Started Guide](https://www.gradio.app/) [cite: 6]
* [Deploying Hugging Face Models + Gradio Tutorial](https://huggingface.co/blog/gradio) [cite: 6]

---

## **Week 7: RAG (Retrieval-Augmented Generation) System Development** [cite: 6]

### 🧠 **Theory** [cite: 6]

* RAG Principles: Retrieval + Generation [cite: 6]
* Vector retrieval libraries (FAISS / Chroma / LlamaIndex) [cite: 6]

### 🛠️ **Practice** [cite: 6]

* Build a local document search with FAISS [cite: 6]
* Build an RAG pipeline with LangChain or LlamaIndex [cite: 6]

### 🔗 **Resources** [cite: 6]

* [Hugging Face RAG Tutorial](https://huggingface.co/docs/transformers/main/en/tasks/retrieval) [cite: 6]
* [Chinese LangChain Tutorial](https://zhuanlan.zhihu.com/p/636471142) [cite: 6]
* [LlamaIndex Getting Started](https://docs.llamaindex.ai/en/stable/) [cite: 6]

---

## **Week 8: Alignment Methods & Model Capability Enhancement** [cite: 6, 7]

### 🧠 **Theory** [cite: 7]

* RLHF vs. DPO [cite: 7]
* Preference data collection and modeling [cite: 7]
* Instruction fine-tuning [cite: 7]

### 🛠️ **Practice** [cite: 7]

* Experiment with `DPOTrainer` for instruction fine-tuning (using `trl`) [cite: 7]
* Read and reproduce the Alpaca/LIMA project's fine-tuning [cite: 7]

### 🔗 **Resources** [cite: 7]

* [DPO Original Paper](https://arxiv.org/abs/2305.18290) [cite: 7]
* [trl: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index) [cite: 7]
* [Alpaca Replication Project](https://github.com/tatsu-lab/stanford_alpaca) [cite: 7]

---

## ✅ **After completion, you will be able to:** [cite: 7]

* Understand the core mechanisms and training methods of LLMs[cite: 7].
* Be proficient in using Hugging Face to train and deploy models[cite: 7].
* Build practical LLM applications like fine-tuning, chatbots, and RAG systems[cite: 7].
* Read mainstream LLM papers and quickly reproduce core technologies[cite: 7].
