
[cite_start]If your ultimate goal is to learn about LLMs (Large Language Models), it's recommended to start with **PyTorch**, as most current LLM research and open-source implementations (like LLaMA, GPT, Mistral, etc.) are based on PyTorch[cite: 1].

[cite_start]Recommended Learning Path (From Scratch to LLM) [cite: 1]

[cite_start]✅ **Phase 1: Master Deep Learning Fundamentals** [cite: 1]
* [cite_start]Start with PyTorch (official tutorials + "Deep Learning from Scratch")[cite: 1].
* [cite_start]Learn the basic principles of neural networks (MLP, CNN, RNN)[cite: 1].
* [cite_start]Get familiar with the Transformer (the core architecture)[cite: 1].

[cite_start]✅ **Phase 2: Learn NLP-related Technologies** [cite: 1]
* [cite_start]Understand tokenization (BPE, WordPiece)[cite: 1].
* [cite_start]Study the Transformer and Self-Attention[cite: 1].
* [cite_start]Practice with small NLP tasks (text classification, named entity recognition, etc.)[cite: 1].

[cite_start]✅ **Phase 3: Dive Deep into LLM Training and Fine-tuning** [cite: 1]
* [cite_start]Study the architectures of GPT, BERT, and LLaMA[cite: 1].
* [cite_start]Learn LoRA and QLoRA (low-cost fine-tuning)[cite: 1].
* [cite_start]Try training or fine-tuning an LLM using Hugging Face Transformers[cite: 1].

[cite_start]✅ **Phase 4: Optimization and Deployment** [cite: 1]
* [cite_start]Understand Quantization and Distillation[cite: 1].
* [cite_start]Try deploying models with vLLM, TGI, and TensorRT-LLM[cite: 1].

---

## [cite_start]**🧠 1. Pre-training Tasks (Masked LM, Causal LM) & Fine-tuning Techniques** [cite: 2]

### [cite_start]✅ **Basic Concepts** [cite: 2]

* [cite_start]**Masked LM (e.g., BERT)**: Randomly masks some tokens in the input and has the model predict them[cite: 2].
* [cite_start]**Causal LM (e.g., GPT)**: Predicts one token at a time by only looking at the left-side context, which is suitable for generation tasks[cite: 2].
* [cite_start]**Fine-tuning Techniques**: Such as full-parameter fine-tuning, Adapter, LoRA, QLoRA, PEFT, etc.[cite: 2].

### [cite_start]📘 **Recommended Resources** [cite: 2]

#### [cite_start]📗 **Beginner (Quickly Grasp Concepts)** [cite: 2]

* [cite_start]***The Illustrated Transformer*** by Jay Alammar [cite: 2]
    [cite_start]👉 [https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/) [cite: 2]

    > [cite_start]Explains the Transformer visually, perfect for beginners[cite: 2].

* [cite_start]**Hugging Face Course** (Recommended!) [cite: 2]
    [cite_start]👉 [https://huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course) [cite: 2]
    * [cite_start]Lesson 3: Pretraining [cite: 2]
    * [cite_start]Lesson 4: Fine-tuning [cite: 2]

    > [cite_start]Designed for actual model training using the `transformers` library, highly practical[cite: 2].

* [cite_start]**Chinese:动手学Transformer** (Hands-on Transformer) [cite: 2]
    [cite_start]👉 GitHub: [https://github.com/datawhalechina/torch-transformers](https://github.com/datawhalechina/torch-transformers) [cite: 2]

    > [cite_start]Covers everything from the Transformer architecture to pre-training tasks and downstream fine-tuning[cite: 2].

#### [cite_start]📘 **Advanced (In-depth Mechanisms and Practice)** [cite: 2]

* [cite_start]**Paper Recommendations**: [cite: 2]

    * [cite_start]BERT Original Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) [cite: 2]
    * [cite_start]GPT Original Paper: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [cite: 2]

* [cite_start]**Practical Recommendations (LoRA/PEFT)** [cite: 2]

    * [cite_start]PEFT Official Documentation: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index) [cite: 2]
    * [cite_start]LoRA [cite: 3] [cite_start]Tutorial Chinese Translation (Zhihu) [cite: 3]
    [cite_start]👉 [https://zhuanlan.zhihu.com/p/610674306](https://zhuanlan.zhihu.com/p/610674306) [cite: 3]

---

## [cite_start]**🎯 2. Prompt Engineering & In-Context Learning Principles** [cite: 3]

### [cite_start]✅ **Core Concepts** [cite: 3]

* [cite_start]**Prompt Engineering**: Using a specific template to guide a model to generate more accurate output[cite: 3].
* [cite_start]**In-Context Learning (ICL)**: Without modifying parameters, you give the model some examples, and it learns how to complete the task[cite: 3].

### [cite_start]📘 **Recommended Resources** [cite: 3]

#### [cite_start]📗 **Beginner Understanding** [cite: 3]

* [cite_start]**Prompt Engineering Guide** (Very comprehensive) [cite: 3]
    [cite_start]👉 [https://github.com/dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) [cite: 3]

    > [cite_start]Covers common strategies like Few-shot, Chain-of-Thought (CoT), ReAct, and is continuously updated[cite: 3].

* [cite_start]**OpenAI Official Tutorial: Prompt Engineering for Developers** [cite: 3]
    [cite_start]👉 [https://platform.openai.com/docs/guides/prompt-engineering](https://platform.openai.com/docs/guides/prompt-engineering) [cite: 3]

    > [cite_start]Designed specifically for ChatGPT/GPT-4 users[cite: 3].

* [cite_start]**Chinese Explanation Videos (Bilibili recommended)** [cite: 3]

    * [cite_start]Search for keywords like "Prompt Engineering 原理" (Prompt Engineering Principles) or "In-Context Learning 机制" (In-Context Learning Mechanisms)[cite: 3]. [cite_start]There are many good NLP bloggers[cite: 3].
    * [cite_start]Recommended: Videos from PaddlePaddle and the Hugging Face Chinese community[cite: 3].

#### [cite_start]📘 **Advanced Resources** [cite: 4]

* [cite_start]**Paper Recommendations**: [cite: 4]

    * [cite_start]GPT-3 Original Paper (Source of ICL theory) [cite: 4]
        [cite_start]👉 [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) [cite: 4]
    * [cite_start]Chain-of-Thought Prompting paper [cite: 4]
        [cite_start]👉 [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903) [cite: 4]

        > [cite_start]Significantly improves performance on complex reasoning problems[cite: 4].

* [cite_start]**Video Course: CS324 @ Stanford (Large [cite: 4] [cite_start]Language Models)** [cite: 4]
    [cite_start]👉 [https://web.stanford.edu/class/cs324/](https://web.stanford.edu/class/cs324/) [cite: 4]

    > [cite_start]Covers the latest cutting-edge topics, including ICL, prompt, alignment, and other core LLM subjects[cite: 4].

---

## [cite_start]🧩 **Suggested Learning Order (Combining both parts)** [cite: 4]

1.  [cite_start]First, use Hugging Face to learn the Transformer architecture, pre-training, and fine-tuning mechanisms[cite: 4].
2.  [cite_start]Then, read the GPT-3 paper to understand what in-context learning is[cite: 4].
3.  [cite_start]Next, practice various prompt engineering techniques (few-shot, CoT, ReAct)[cite: 4].
4.  [cite_start]Finally, dive into efficient fine-tuning methods like LoRA/QLoRA[cite: 4].

---

This is great! [cite_start]Here is a customized **8-Week LLM Learning Plan** for you, with the goal of: mastering the core principles of LLMs, fine-tuning methods, Prompt Engineering, and inference mechanisms, ultimately being able to independently build and deploy an LLM application[cite: 4].

---

# [cite_start]🧭 **8-Week LLM Learning Plan (Engineering + Research Focus)** [cite: 4]

### [cite_start]👨‍💻 **Approximate Weekly Time Commitment: 8-12 hours (Flexible)** [cite: 4]

[cite_start]Each week includes three parts: [cite: 4]

* [cite_start]🧠 Theory Learning [cite: 4]
* [cite_start]🛠️ Hands-on Practice [cite: 4]
* [cite_start]🔗 Resource Links [cite: 4]

---

## [cite_start]**Week 1: Transformer Architecture & Pre-training Concepts** [cite: 4]

### [cite_start]🧠 **Theory** [cite: 4]

* [cite_start]Transformer Architecture and Attention [cite: 4]
* [cite_start]Masked LM vs. Causal LM [cite: 4]
* [cite_start]Difference between Pre-training and Downstream Tasks [cite: 4]

### [cite_start]🛠️ **Practice** [cite: 4]

* [cite_start]Load BERT and GPT-2 with `transformers` [cite: 4]
* [cite_start]Use GPT-2 to generate text and experience causal LM [cite: 4]

### [cite_start]🔗 **Resources** [cite: 4]

* [cite_start][The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) [cite: 4]
* [cite_start][Hugging Face NLP Course Chapter 1–4](https://huggingface.co/learn/nlp-course) [cite: 4]
* [cite_start][Transformers Documentation Introduction](https://huggingface.co/docs/transformers/index) [cite: 4]

---

## [cite_start]**Week 2: Hugging [cite: 5] [cite_start]Face Practice + Fine-tuning BERT on Custom Data** [cite: 5]

### [cite_start]🧠 **Theory** [cite: 5]

* [cite_start]How Tokenizers Work [cite: 5]
* [cite_start]Basic Fine-tuning Workflow [cite: 5]

### [cite_start]🛠️ **Practice** [cite: 5]

* [cite_start]Fine-tune BERT for sentiment analysis or text classification (IMDb dataset) [cite: 5]
* [cite_start]Use Datasets + Trainer API [cite: 5]

### [cite_start]🔗 **Resources** [cite: 5]

* [cite_start][Hugging Face Docs - Fine-tuning BERT](https://huggingface.co/docs/transformers/training) [cite: 5]
* [cite_start][IMDb Dataset](https://huggingface.co/datasets/imdb) [cite: 5]
* [cite_start][Official Fine-tuning Tutorial Notebook](https://github.com/huggingface/notebooks/blob/main/course/en/chapter3/section3.ipynb) [cite: 5]

---

## [cite_start]**Week 3: Causal LM Training + GPT-2 Text Generation Techniques** [cite: 5]

### [cite_start]🧠 **Theory** [cite: 5]

* [cite_start]Decoding Strategies (Greedy, Beam, Top-k, Top-p) [cite: 5]
* [cite_start]Meaning of Temperature and Repetition Penalty [cite: 5]

### [cite_start]🛠️ **Practice** [cite: 5]

* [cite_start]Use GPT-2 for text generation tasks (e.g., writing poems, dialogues) [cite: 5]
* [cite_start]Compare the effects of different decoding strategies [cite: 5]

### [cite_start]🔗 **Resources** [cite: 5]

* [cite_start][Detailed Blog on Text Generation](https://huggingface.co/blog/how-to-generate) [cite: 5]
* [cite_start][GPT2 Text Generation Notebook Example](https://github.com/huggingface/notebooks/blob/main/examples/text_generation.ipynb) [cite: 5]

---

## [cite_start]**Week 4: Prompt Engineering & In-Context Learning** [cite: 5]

### [cite_start]🧠 **Theory** [cite: 5]

* [cite_start]Zero-shot / Few-shot Prompting [cite: 5]
* [cite_start]Chain-of-Thought (CoT) Prompting [cite: 5]
* [cite_start]Introduction to ReAct Models [cite: 5]

### [cite_start]🛠️ **Practice** [cite: 5]

* [cite_start]Construct prompts to perform sentiment classification and solve logic problems [cite: 5]
* [cite_start]Compare performance with and without CoT [cite: 5]

### [cite_start]🔗 **Resources** [cite: 5]

* [cite_start][Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) [cite: 5]
* [cite_start][CoT Prompting Paper](https://arxiv.org/abs/2201.11903) [cite: 5]
* [cite_start][OpenAI Prompt Guide](https://platform.openai.com/docs/guides/prompt-engineering) [cite: 5]

---

## [cite_start]**Week 5: LoRA / PEFT Fine-tuning Techniques** [cite: 5]

### [cite_start]🧠 **Theory** [cite: 6]

* [cite_start]Full-parameter fine-tuning [cite: 6] [cite_start]vs. LoRA/Adapter [cite: 6]
* [cite_start]PEFT core idea: freezing the large model and only fine-tuning a few layers [cite: 6]

### [cite_start]🛠️ **Practice** [cite: 6]

* [cite_start]Fine-tune LLaMA/Mistral/ChatGLM models using `peft` [cite: 6]
* [cite_start]Try using QLoRA to train models in a low-VRAM environment [cite: 6]

### [cite_start]🔗 **Resources** [cite: 6]

* [cite_start][PEFT Official Documentation](https://huggingface.co/docs/peft) [cite: 6]
* [cite_start][QLoRA Original Paper](https://arxiv.org/abs/2305.14314) [cite: 6]
* [cite_start][Chinese QLoRA Tutorial](https://zhuanlan.zhihu.com/p/640898922) [cite: 6]

---

## [cite_start]**Week 6: LLM Application Development + Gradio Frontend** [cite: 6]

### [cite_start]🧠 **Theory** [cite: 6]

* [cite_start]How to deploy a fine-tuned model [cite: 6]
* [cite_start]Gradio/Streamlit frontend frameworks [cite: 6]

### [cite_start]🛠️ **Practice** [cite: 6]

* [cite_start]Build a Chatbot interface [cite: 6]
* [cite_start]Support prompt input, setting temperature, max tokens, and other parameters [cite: 6]

### [cite_start]🔗 **Resources** [cite: 6]

* [cite_start][Gradio Getting Started Guide](https://www.gradio.app/) [cite: 6]
* [cite_start][Deploying Hugging Face Models + Gradio Tutorial](https://huggingface.co/blog/gradio) [cite: 6]

---

## [cite_start]**Week 7: RAG (Retrieval-Augmented Generation) System Development** [cite: 6]

### [cite_start]🧠 **Theory** [cite: 6]

* [cite_start]RAG Principles: Retrieval + Generation [cite: 6]
* [cite_start]Vector retrieval libraries (FAISS / Chroma / LlamaIndex) [cite: 6]

### [cite_start]🛠️ **Practice** [cite: 6]

* [cite_start]Build a local document search with FAISS [cite: 6]
* [cite_start]Build an RAG pipeline with LangChain or LlamaIndex [cite: 6]

### [cite_start]🔗 **Resources** [cite: 6]

* [cite_start][Hugging Face RAG Tutorial](https://huggingface.co/docs/transformers/main/en/tasks/retrieval) [cite: 6]
* [cite_start][Chinese LangChain Tutorial](https://zhuanlan.zhihu.com/p/636471142) [cite: 6]
* [cite_start][LlamaIndex Getting Started](https://docs.llamaindex.ai/en/stable/) [cite: 6]

---

## [cite_start]**Week 8: Alignment Methods & Model Capability Enhancement** [cite: 6, 7]

### [cite_start]🧠 **Theory** [cite: 7]

* [cite_start]RLHF vs. DPO [cite: 7]
* [cite_start]Preference data collection and modeling [cite: 7]
* [cite_start]Instruction fine-tuning [cite: 7]

### [cite_start]🛠️ **Practice** [cite: 7]

* [cite_start]Experiment with `DPOTrainer` for instruction fine-tuning (using `trl`) [cite: 7]
* [cite_start]Read and reproduce the Alpaca/LIMA project's fine-tuning [cite: 7]

### [cite_start]🔗 **Resources** [cite: 7]

* [cite_start][DPO Original Paper](https://arxiv.org/abs/2305.18290) [cite: 7]
* [cite_start][trl: Transformer Reinforcement Learning](https://huggingface.co/docs/trl/index) [cite: 7]
* [cite_start][Alpaca Replication Project](https://github.com/tatsu-lab/stanford_alpaca) [cite: 7]

---

## [cite_start]✅ **After completion, you will be able to:** [cite: 7]

* [cite_start]Understand the core mechanisms and training methods of LLMs[cite: 7].
* [cite_start]Be proficient in using Hugging Face to train and deploy models[cite: 7].
* [cite_start]Build practical LLM applications like fine-tuning, chatbots, and RAG systems[cite: 7].
* [cite_start]Read mainstream LLM papers and quickly reproduce core technologies[cite: 7].
