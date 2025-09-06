If your ultimate goal is to learn about LLMs (Large Language Models), it's recommended to start with PyTorch because most LLM research and open-source implementations (like LLaMA, GPT, Mistral, etc.) are currently based on PyTorch.

Here's a recommended learning path (from scratch to LLM):

## Learning Path Recommendation (From Zero to LLM)

**Phase 1: Master Deep Learning Fundamentals**
* First, learn PyTorch (official tutorials + "Deep Learning from Scratch").
* Understand the basic principles of neural networks (MLP, CNN, RNN).
* Familiarize yourself with the Transformer architecture (the core).

**Phase 2: Learn NLP-Related Technologies**
* Understand tokenization (BPE, WordPiece).
* Study Transformers and Self-Attention.
* Practice with small NLP tasks (text classification, named entity recognition, etc.).

**Phase 3: Dive into LLM Training and Fine-tuning**
* Study the architectures of GPT, BERT, and LLaMA.
* Learn LoRA, QLoRA (low-cost fine-tuning methods).
* Try training or fine-tuning LLMs using Hugging Face Transformers.

**Phase 4: Optimization and Deployment**
* Learn about Quantization and Distillation.
* Experiment with deploying models using vLLM, TGI, or TensorRT-LLM.

## Recommended Resources

### Beginner (Quick Concept Understanding)

* **"The Illustrated Transformer" by Jay Alammar**
    👉 [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
    * Explains the Transformer architecture visually, making it ideal for beginners.

* **Hugging Face Course (Highly Recommended!)**
    👉 [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)
    * Lesson 3: Pretraining
    * Lesson 4: Fine-tuning
    * Designed for practical model training using the `transformers` library, offering high utility.

* **In Chinese: "Hands-on Transformer" (Simplified Chinese Project)**
    👉 GitHub: [https://github.com/datawhalechina/torch-transformers](https://github.com/datawhalechina/torch-transformers)
    * Covers everything from the Transformer architecture to pre-training tasks and downstream fine-tuning.

### Advanced (In-depth Mechanisms and Practice)

* **Recommended Papers:**
    * BERT Original Paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
    * GPT Original Paper: Language Models are Unsupervised Multitask Learners

* **Practical Recommendations (LoRA/PEFT):**
    * PEFT Official Documentation: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
    * LoRA Tutorial Chinese Translation (Zhihu):
        👉 [https://zhuanlan.zhihu.com/p/610674306](https://zhuanlan.zhihu.com/p/610674306)

## Suggested Learning Sequence (Combining Both Parts)

1.  First, use Hugging Face to learn the Transformer architecture, pre-training, and fine-tuning mechanisms.
2.  Then, read the GPT-3 paper to understand in-context learning.
3.  Next, practice various prompt engineering techniques (few-shot, CoT, ReAct).
4.  Finally, delve into efficient fine-tuning methods like LoRA/QLoRA.

---

These are two excellent learning directions! Here are high-quality learning paths and recommended resources for the two themes you mentioned (primarily English resources, with some interspersed Chinese video/blog explanations):

---

## 🧠 1. **Pre-training Tasks (Masked LM, Causal LM) and Fine-tuning Techniques**

### ✅ Basic Concepts

* **Masked LM (e.g., BERT):** Randomly masks some tokens in the input and expects the model to predict them.
* **Causal LM (e.g., GPT):** Looks only at the left context and predicts tokens one by one, suitable for generation tasks.
* **Fine-tuning Techniques:** Includes full parameter fine-tuning, Adapters, LoRA, QLoRA, PEFT, etc.

### 📘 Recommended Resources

#### 📗 Beginner (Quick Concept Understanding)

* **"The Illustrated Transformer" by Jay Alammar**
    👉 [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
    > Explains the Transformer visually, making it very suitable for beginners.

* **Hugging Face Course** (Recommended!)
    👉 [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)
    * Lesson 3: Pretraining
    * Lesson 4: Fine-tuning
    > Completely designed for practical model training using the `transformers` library, offering high utility.

* **In Chinese: "Hands-on Transformer"** (Simplified Chinese Project)
    👉 GitHub: [https://github.com/datawhalechina/torch-transformers](https://github.com/datawhalechina/torch-transformers)
    > Covers everything from the Transformer architecture to pre-training tasks and downstream fine-tuning.

#### 📘 Advanced (In-depth Mechanisms and Practice)

* **Recommended Papers:**
    * BERT Original Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
    * GPT Original Paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

* **Practical Recommendations (LoRA/PEFT)**
    * PEFT Official Documentation: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
    * LoRA Tutorial Chinese Translation (Zhihu):
        👉 [https://zhuanlan.zhihu.com/p/610674306](https://zhuanlan.zhihu.com/p/610674306)

---

## 🎯 2. **Prompt Engineering & In-Context Learning Principles**

### ✅ Core Concepts

* **Prompt Engineering:** Using specific templates to guide the model to generate more accurate outputs.
* **In-Context Learning (ICL):** Without modifying parameters, providing the model with a few examples allows it to learn how to complete tasks.

### 📘 Recommended Resources

#### 📗 Beginner Understanding

* **Prompt Engineering Guide** (Comprehensive)
    👉 [https://github.com/dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
    > Covers common strategies like Few-shot, Chain-of-Thought (CoT), ReAct, and is continuously updated.

* **OpenAI Official Tutorial: Prompt Engineering for Developers**
    👉 [https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering)
    > Specifically designed for ChatGPT/GPT-4 users.

* **Chinese Explanation Videos (Recommended on Bilibili)**
    * Search for keywords like "Prompt Engineering Principles," "In-Context Learning Mechanism," etc. Many NLP bloggers provide good explanations.
    * Recommended: Videos from PaddlePaddle and Hugging Face Chinese Community.

#### 📘 Advanced Resources

* **Recommended Papers:**
    * GPT-3 Original Paper (Theoretical basis for ICL):
        👉 [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
    * Chain-of-Thought Prompting Paper:
        👉 [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
        > Significantly improves performance on complex reasoning tasks.

* **Video Course: CS324 @ Stanford (Large Language Models)**
    👉 [https://web.stanford.edu/class/cs324/](https://web.stanford.edu/class/cs324/)
    > Latest cutting-edge content, covering core LLM topics like ICL, prompts, and alignment.

---

## 🧩 Suggested Learning Sequence (Combining Both Parts)

1.  First, use Hugging Face to learn the Transformer architecture, pre-training, and fine-tuning mechanisms.
2.  Then, read the GPT-3 paper to understand in-context learning.
3.  Next, practice various prompt engineering techniques (few-shot, CoT, ReAct).
4.  Finally, delve into efficient fine-tuning methods like LoRA/QLoRA.

---

Great, here is a tailor-made **8-Week LLM Study Plan** designed to help you master the core principles of LLMs, fine-tuning methods, prompt engineering, inference mechanisms, and ultimately enable you to independently build and deploy an LLM application.

---

# 🧭 8-Week LLM Study Plan (Engineering + Research Track)

### 👨‍💻 Approximate Weekly Commitment: 8-12 hours (flexible)

Each week includes three parts:

* 🧠 Theoretical Learning
* 🛠️ Practical Application
* 🔗 Resource Links

---

## **Week 1: Transformer Architecture and Pre-training Concepts**

### 🧠 Theory

* Transformer Architecture and Attention
* Masked LM vs. Causal LM
* Distinction between Pre-training and Downstream Tasks

### 🛠️ Practice

* Load BERT and GPT-2 using `transformers`.
* Generate text with GPT-2 to experience Causal LM.

### 🔗 Resource Links

* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
* [Hugging Face NLP Course Chapters 1-4](https://huggingface.co/learn/nlp-course)
* [Transformers Documentation - Getting Started](https://huggingface.co/docs/transformers/index)

---

## **Week 2: Hugging Face Practical Use + Fine-tuning BERT with Custom Data**

### 🧠 Theory

* How Tokenizers Work
* Fine-tuning Basics

### 🛠️ Practice

* Fine-tune BERT for sentiment analysis or text classification (IMDb dataset).
* Use the `Datasets` + `Trainer` API.

### 🔗 Resource Links

* [Hugging Face Documentation - Fine-tuning BERT](https://huggingface.co/docs/transformers/training)
* [IMDb Dataset](https://huggingface.co/datasets/imdb)
* [Official Fine-tuning Tutorial Notebook](https://github.com/huggingface/notebooks/blob/main/course/en/chapter3/section3.ipynb)

---

## **Week 3: Causal LM Training + GPT-2 Text Generation Techniques**

### 🧠 Theory

* Decoding Strategies (Greedy, Beam, Top-k, Top-p)
* Meaning of Temperature and Repetition Penalty

### 🛠️ Practice

* Use GPT-2 for text generation tasks (e.g., writing poetry, dialogue).
* Compare the effects of different decoding strategies.

### 🔗 Resource Links

* [Blog Post on Text Generation](https://huggingface.co/blog/how-to-generate)
* [GPT-2 Text Generation Notebook Example](https://github.com/huggingface/notebooks/blob/main/examples/text_generation.ipynb)

---

## **Week 4: Prompt Engineering & In-Context Learning**

### 🧠 Theory

* Zero-shot / Few-shot Prompts
* Chain-of-Thought (CoT) Prompting
* Introduction to the ReAct Model

### 🛠️ Practice

* Construct prompts to achieve sentiment classification and logical problem-solving.
* Compare performance with and without CoT.

### 🔗 Resource Links

* [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
* [CoT Prompting Paper](https://arxiv.org/abs/2201.11903)
* [OpenAI Prompt Guide](https://platform.openai.com/docs/guides/prompt-engineering)

---

## **Week 5: LoRA / PEFT Fine-tuning Techniques**

### 🧠 Theory

* Full Parameter Fine-tuning vs. LoRA/Adapters
* Core idea of PEFT: Freeze the large model, fine-tune only a few layers.

### 🛠️ Practice

* Fine-tune LLaMA/Mistral/ChatGLM models using `peft`.
* Attempt to train models in low-memory environments using QLoRA.

### 🔗 Resource Links

* [PEFT Official Documentation](https://huggingface.co/docs/peft)
* [QLoRA Original Paper](https://arxiv.org/abs/2305.14314)
* [Chinese QLoRA Tutorial](https://zhuanlan.zhihu.com/p/640898922)

---

## **Week 6: LLM Application Development + Gradio Frontend**

### 🧠 Theory

* How to deploy fine-tuned models.
* Frontend frameworks like Gradio/Streamlit.

### 🛠️ Practice

* Build a Chatbot interface.
* Support prompt input, temperature settings, max tokens, and other parameters.

### 🔗 Resource Links

* [Gradio Getting Started](https://www.gradio.app/)
* [Tutorial on Deploying Hugging Face Models with Gradio](https://huggingface.co/blog/gradio)

---

## **Week 7: RAG (Retrieval-Augmented Generation) System Development**

### 🧠 Theory

* RAG Principles: Search + Generation
* Vector Databases (FAISS / Chroma / LlamaIndex)

### 🛠️ Practice

* Build a local document search using FAISS.
* Construct a RAG pipeline using LangChain or LlamaIndex.

### 🔗 Resource Links

* [Hugging Face RAG Tutorial](https://huggingface.co/docs/transformers/main/en/tasks/retrieval)
* [LangChain Chinese Tutorial](https://zhuanlan.zhihu.com/p/636471142)
* [LlamaIndex Getting Started](https://docs.llamaindex.ai/en/stable/)

---

## **Week 8: Alignment Methods (Alignment) and Model Capability Improvement**

### 🧠 Theory

* RLHF vs. DPO
* Preference Data Collection and Modeling
* Instruction Tuning

### 🛠️ Practice

* Experiment with `DPOTrainer` for instruction tuning (using `trl`).
* Read and replicate the fine-tuning of Alpaca/LIMA projects.

### 🔗 Resource Links

* [DPO Original Paper](https://arxiv.org/abs/2305.18290)
* [trl: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index)
* [Alpaca Replication Project](https://github.com/tatsu-lab/stanford_alpaca)

---

## ✅ After Completion, You Will Be Able To:

* Understand the core mechanisms and training methods of LLMs.
* Proficiently use Hugging Face for training and deploying models.
* Build practical LLM applications such as chatbots and RAG systems.
* Read mainstream LLM papers and quickly replicate core technologies.

---
