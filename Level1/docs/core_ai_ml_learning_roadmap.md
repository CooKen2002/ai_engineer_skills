# Core AI/ML Knowledge — Learning Roadmap

> Mục tiêu: xây nền tảng AI/ML đủ chắc để **hiểu bản chất model**, đọc paper, tự implement các thành phần cốt lõi và sau đó mới dùng framework/library như PyTorch, Hugging Face, vector database, RAG...
>
> Định hướng phù hợp cho NLP / Speech / LLM / RAG:
>
> **Math → Machine Learning → Neural Networks → CNN/RNN → Attention → Transformer → LLM → Embedding → Vector Search → RAG**

---

## Mục lục

1. [Mục tiêu học tập](#1-mục-tiêu-học-tập)
2. [Learning path tổng thể](#2-learning-path-tổng-thể)
3. [Phase 0 — Mathematical Foundations](#3-phase-0--mathematical-foundations)
4. [Phase 1 — Machine Learning cơ bản](#4-phase-1--machine-learning-cơ-bản)
5. [Phase 2 — Neural Networks & Deep Learning](#5-phase-2--neural-networks--deep-learning)
6. [Phase 3 — PyTorch](#6-phase-3--pytorch)
7. [Phase 4 — CNN](#7-phase-4--cnn)
8. [Phase 5 — RNN / LSTM](#8-phase-5--rnn--lstm)
9. [Phase 6 — Attention & Transformer](#9-phase-6--attention--transformer)
10. [Phase 7 — LLM Fundamentals](#10-phase-7--llm-fundamentals)
11. [Phase 8 — Embedding & Vector Search](#11-phase-8--embedding--vector-search)
12. [Phase 9 — RAG Fundamentals](#12-phase-9--rag-fundamentals)
13. [Bộ tài liệu must-read](#13-bộ-tài-liệu-must-read)
14. [Learning projects](#14-learning-projects)
15. [Questions để tự kiểm tra](#15-questions-để-tự-kiểm-tra)
16. [Nguyên tắc học](#16-nguyên-tắc-học)
17. [Roadmap ngắn gọn](#17-roadmap-ngắn-gọn)

---

# 1. Mục tiêu học tập

Sau roadmap này, mục tiêu là có thể:

- Hiểu Machine Learning fundamentals.
- Phân biệt Regression / Classification / Clustering.
- Hiểu precision / recall / F1 / confusion matrix / threshold.
- Hiểu loss function, gradient descent, backpropagation và optimization.
- Tự implement một neural network nhỏ từ đầu.
- Sử dụng **PyTorch** ở mức hiểu được training loop và autograd bên dưới.
- Hiểu CNN về mặt kiến trúc, convolution, receptive field, parameter sharing.
- Hiểu RNN / LSTM và vấn đề long-term dependency.
- Hiểu Attention và Transformer từ công thức đến architecture.
- Hiểu token, tokenizer, embedding, logits, softmax, sampling, temperature, context window.
- Hiểu LLM sinh text như thế nào.
- Hiểu embedding và semantic similarity.
- Hiểu vector search / ANN / FAISS.
- Hiểu RAG từ chunking → embedding → retrieval → reranking → generation.
- Có khả năng đọc paper cơ bản thay vì chỉ gọi API.

---

# 2. Learning path tổng thể

```text
PHASE 0
Math fundamentals
    │
    ▼
PHASE 1
Machine Learning
- Linear Regression
- Logistic Regression
- Classification
- Precision / Recall / F1
- Overfitting
- Regularization
    │
    ▼
PHASE 2
Neural Networks
- Forward pass
- Loss
- Backpropagation
- Gradient Descent
    │
    ▼
PHASE 3
PyTorch
- Tensor
- Dataset / DataLoader
- Model
- Autograd
- Optimizer
- Training Loop
    │
    ├──────────────► PHASE 4: CNN
    │
    └──────────────► PHASE 5: RNN / LSTM
                              │
                              ▼
                       PHASE 6: Attention
                              │
                              ▼
                         Transformer
                              │
                              ▼
                        PHASE 7: LLM
                    - Tokenization
                    - Embedding
                    - Causal LM
                    - Logits
                    - Softmax
                    - Sampling
                    - Temperature
                    - Context Window
                              │
                              ▼
                  PHASE 8: Embedding
                    - Vector representation
                    - Similarity
                    - Vector Search
                    - ANN / FAISS
                              │
                              ▼
                    PHASE 9: RAG
                    - Chunking
                    - Retrieval
                    - Reranking
                    - Generation
                    - Evaluation
```

---

# 3. Phase 0 — Mathematical Foundations

Không cần trở thành mathematician, nhưng cần đủ nền tảng để hiểu model.

## 3.1 Linear Algebra

Cần biết:

- Vector
- Matrix
- Dot product
- Matrix multiplication
- Norm
- Projection
- Eigenvalue / Eigenvector

Đặc biệt với Transformer cần hiểu:

```text
Q = XWq
K = XWk
V = XWv
```

và tại sao matrix multiplication lại được dùng để biến đổi representation.

## 3.2 Calculus

Cần biết:

- Derivative
- Partial derivative
- Gradient
- Chain rule

Mục tiêu cuối cùng:

```text
Loss
  ↓
∂Loss / ∂parameter
  ↓
Gradient
  ↓
Parameter update
```

## 3.3 Probability

Cần biết:

- Probability distribution
- Expectation
- Variance
- Conditional probability
- Bayes theorem
- Maximum likelihood

## 3.4 Optimization

Cần hiểu:

- Loss function
- Gradient descent
- Learning rate
- Local / global minima
- Regularization
- SGD
- Adam

## Tài liệu

### Dive into Deep Learning

https://www.d2l.ai/

Đây là resource rất tốt vì kết hợp:

- Mathematics
- Theory
- Visualizations
- Jupyter notebooks
- PyTorch implementation

---

# 4. Phase 1 — Machine Learning cơ bản

## 4.1 Các concept phải nắm

### Regression

Mục tiêu:

```text
Input → Continuous value
```

Ví dụ:

```text
house features → house price
```

Linear regression:

```text
y = wx + b
```

Cần hiểu:

- MSE
- Least squares
- Gradient
- Gradient descent
- Bias / variance

### Classification

Mục tiêu:

```text
Input → Class
```

Ví dụ:

```text
email → spam / not spam
```

Logistic regression:

```text
z = wx + b
p = sigmoid(z)
```

### Evaluation

Bắt buộc hiểu:

- Confusion Matrix
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

Precision:

```text
precision = TP / (TP + FP)
```

Recall:

```text
recall = TP / (TP + FN)
```

F1:

```text
F1 = 2 * precision * recall / (precision + recall)
```

Không chỉ nhớ công thức. Phải hiểu:

> Khi nào cần ưu tiên precision?

> Khi nào cần ưu tiên recall?

> Vì sao tăng threshold có thể làm precision tăng nhưng recall giảm?

### Các topic tiếp theo

- Train / validation / test split
- Cross-validation
- Bias / variance
- Overfitting
- Underfitting
- Regularization
- L1 / L2
- Decision trees
- Random forests
- SVM
- Unsupervised learning

---

## 4.2 ⭐ Must-read — An Introduction to Statistical Learning with Python

https://www.statlearning.com/home

Đây là cuốn sách rất tốt để xây ML foundation.

### Ưu tiên đọc

```text
Introduction
Linear Regression
Classification
Resampling
Regularization
Deep Learning
Unsupervised Learning
```

Các chapter như Tree-Based Methods và SVM có thể học sau.

---

## 4.3 ⭐ Google Machine Learning Crash Course

https://developers.google.com/machine-learning/crash-course

Có:

- ML fundamentals
- Linear regression
- Logistic regression
- Classification
- Overfitting
- Neural networks
- Embeddings
- Intro to LLMs
- Exercises trên Colab

### Classification metrics

https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

Đặc biệt tập trung:

```text
TP / TN / FP / FN
Precision
Recall
F1
Threshold
ROC-AUC
```

---

# 5. Phase 2 — Neural Networks & Deep Learning

## 5.1 Neural Network cơ bản

Cần hiểu:

```text
Input
  ↓
Linear / Dense
  ↓
Activation
  ↓
Hidden layers
  ↓
Output
```

Một neuron:

```text
z = w1*x1 + w2*x2 + ... + b
a = activation(z)
```

Các activation quan trọng:

- ReLU
- Sigmoid
- Tanh
- Softmax

## 5.2 Training process

Một training loop cơ bản:

```text
Input
  ↓
Forward pass
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Update parameters
  ↓
Repeat
```

Đây là flow phải hiểu thật rõ trước khi học LLM.

---

## 5.3 ⭐ Andrej Karpathy — Neural Networks: Zero to Hero

https://github.com/karpathy/nn-zero-to-hero

Một trong những resource tốt nhất để hiểu neural network từ code.

Các concept:

```text
Neuron
↓
Forward pass
↓
Loss
↓
Backpropagation
↓
Gradient descent
↓
Neural network
↓
Language model
↓
Transformer
```

### micrograd

Cực kỳ đáng học để hiểu autograd / backpropagation.

Mục tiêu:

```text
y = f(x)
     ↓
loss
     ↓
∂loss/∂x
     ↓
gradient
     ↓
update
```

---

## 5.4 ⭐ Dive into Deep Learning

https://www.d2l.ai/

Các chapter ưu tiên:

```text
2. Preliminaries
3. Linear Neural Networks
4. Multilayer Perceptrons
5. Deep Learning Computation
6. Convolutional Neural Networks
9. Modern Convolutional Neural Networks
10. Recurrent Neural Networks
11. Modern Recurrent Neural Networks
12. Attention Mechanisms and Transformers
```

---

## 5.5 Deep Learning Book

**Deep Learning — Goodfellow, Bengio, Courville**

https://www.deeplearningbook.org/

Không nhất thiết đọc cover-to-cover ngay.

Dùng như reference khi cần đào sâu:

- Optimization
- Regularization
- Representation learning
- CNN
- Sequence models
- Backpropagation

---

# 6. Phase 3 — PyTorch

## Lựa chọn framework

Chọn:

# PyTorch

Không cần học cả TensorFlow và PyTorch cùng lúc.

---

## 6.1 PyTorch official tutorial

https://docs.pytorch.org/tutorials/beginner/basics/intro.html

Learning path:

```text
Tensor
↓
Dataset / DataLoader
↓
Build Model
↓
Autograd
↓
Optimization
↓
Save / Load
```

---

## 6.2 Những thứ phải hiểu

Không chỉ biết API.

Phải hiểu:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

thực chất đang làm:

```text
forward
↓
compute loss
↓
backpropagation
↓
compute gradients
↓
update parameters
↓
clear gradients
```

---

## 6.3 Training loop

Phải tự viết được dạng:

```python
for x, y in dataloader:
    optimizer.zero_grad()

    prediction = model(x)
    loss = loss_fn(prediction, y)

    loss.backward()
    optimizer.step()
```

và giải thích từng dòng.

---

# 7. Phase 4 — CNN

## 7.1 Stanford CS231n

https://cs231n.stanford.edu/

Resource kinh điển cho CNN / Computer Vision.

---

## 7.2 Kiến trúc CNN

```text
Image
 ↓
Convolution
 ↓
Activation
 ↓
Pooling / Stride
 ↓
Feature maps
 ↓
Flatten / Global pooling
 ↓
Linear
 ↓
Prediction
```

Phải hiểu:

- Convolution
- Kernel / filter
- Stride
- Padding
- Channels
- Feature maps
- Receptive field
- Parameter sharing

### Câu hỏi quan trọng

> Tại sao CNN phù hợp cho image?

> Tại sao convolution giảm số parameter so với fully connected?

> Parameter sharing là gì?

> Receptive field là gì?

---

## 7.3 ⭐ ResNet paper

**Deep Residual Learning for Image Recognition**

https://arxiv.org/abs/1512.03385

Concept chính:

```text
x → F(x)

thay vì:

x → F(x) + x
```

Residual connection giúp train mạng sâu hiệu quả hơn.

---

# 8. Phase 5 — RNN / LSTM

Dù modern LLM chủ yếu dùng Transformer, RNN/LSTM vẫn rất đáng học vì giúp hiểu:

> Transformer xuất hiện để giải quyết những giới hạn nào?

---

## 8.1 RNN

Concept:

```text
x1 → RNN → h1
x2 → RNN → h2
x3 → RNN → h3
...
```

Công thức khái niệm:

```text
h_t = f(x_t, h_(t-1))
```

Phải hiểu:

- Hidden state
- Sequence modeling
- Vanishing gradient
- Exploding gradient
- Long-term dependency

---

## 8.2 ⭐ LSTM paper

**Long Short-Term Memory — Hochreiter & Schmidhuber**

https://doi.org/10.1162/neco.1997.9.8.1735

Phải hiểu:

- Cell state
- Forget gate
- Input gate
- Output gate

Mục tiêu:

```text
RNN
↓
khó nhớ long-term dependency
↓
LSTM
↓
gated memory
```

---

# 9. Phase 6 — Attention & Transformer

Đây là phase quan trọng nhất để bước vào LLM.

---

## 9.1 ⭐⭐⭐ Attention Is All You Need

Paper:

https://arxiv.org/abs/1706.03762

Đây là paper gốc của Transformer.

---

## 9.2 Kiến trúc Transformer

Concept tổng quát:

```text
Input tokens
      ↓
Token Embedding
      ↓
Positional information
      ↓
Self-Attention
      ↓
Multi-Head Attention
      ↓
Feed Forward Network
      ↓
Residual + LayerNorm
      ↓
Output
```

---

## 9.3 Self-attention

Phải tự hiểu công thức:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
= softmax(QKᵀ / √d)V
```

Cần trả lời:

### Q là gì?

Query.

### K là gì?

Key.

### V là gì?

Value.

### Tại sao có `QKᵀ`?

Để tính mức độ tương thích giữa query và key.

### Tại sao chia `sqrt(d_k)`?

Để scale dot products và giúp softmax / optimization ổn định hơn.

### Tại sao Multi-Head Attention?

Để các head có thể học những kiểu quan hệ khác nhau trong sequence.

---

## 9.4 Causal Attention

Đối với autoregressive language model:

```text
token hiện tại
    ↓
chỉ được nhìn token trước nó
```

nên dùng causal mask.

Ví dụ:

```text
A B C D

A → A
B → A B
C → A B C
D → A B C D
```

---

# 10. Phase 7 — LLM Fundamentals

## 10.1 Tokenization

Pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding vectors
 ↓
Transformer
```

Token không nhất thiết bằng word.

Một word có thể:

```text
word
→ one token
```

hoặc:

```text
word
→ multiple subword tokens
```

---

## 10.2 Hugging Face Tokenizers

https://huggingface.co/docs/tokenizers/

Pipeline thường gồm:

```text
Normalization
↓
Pre-tokenization
↓
Tokenization model
↓
Post-processing
```

Các tokenizer hiện đại thường sử dụng subword techniques như BPE.

---

## 10.3 Embedding

Ví dụ khái niệm:

```text
"cat"
→ [0.21, -0.31, ...]
```

Embedding là vector representation của text/token/object.

Cần phân biệt:

```text
Token ID
≠
Embedding vector
```

---

## 10.4 Causal Language Model

LLM autoregressive thực hiện:

```text
"The cat is"
      ↓
predict next token
      ↓
"sitting"
```

rồi:

```text
"The cat is sitting"
      ↓
predict next token
```

và lặp lại.

---

## 10.5 Logits

Model thường tạo logits:

```text
logits
↓
softmax
↓
probability distribution
↓
decoding / sampling
```

Model không trực tiếp "chọn câu trả lời".

Nó tạo phân phối xác suất cho token tiếp theo.

---

## 10.6 Sampling

Các chiến lược cần biết:

- Greedy decoding
- Temperature sampling
- Top-k
- Top-p / nucleus sampling
- Beam search (đặc biệt trong seq2seq)

---

## 10.7 Temperature

Concept:

```text
logits
  ↓
divide by temperature
  ↓
softmax
  ↓
probabilities
  ↓
sampling
```

Nhiệt độ thấp:

```text
distribution tập trung hơn
```

Nhiệt độ cao:

```text
distribution phẳng hơn
```

Đừng chỉ ghi nhớ:

> "temperature thấp = deterministic"

Quan trọng là phải hiểu tác động của temperature lên logits trước softmax.

---

## 10.8 Context Window

Hiểu:

- Input context
- Output tokens
- Maximum context
- Token budget
- Context management

Điều này đặc biệt quan trọng khi xây RAG.

---

## 10.9 ⭐ Hugging Face LLM Course

https://huggingface.co/learn/llm-course/en/chapter1/1

Nội dung:

- Transformers
- Tokenizers
- Pretrained models
- Datasets
- Fine-tuning
- Causal language modeling
- Inference
- Advanced LLM topics

---

## 10.10 ⭐ GPT-2 paper

**Language Models are Unsupervised Multitask Learners**

https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Giúp nối:

```text
Transformer
↓
GPT architecture
↓
Autoregressive language modeling
↓
Large-scale language model
```

Paper cũng hữu ích để hiểu representation / tokenization / BPE.

---

# 11. Phase 8 — Embedding & Vector Search

## 11.1 Semantic Embedding

Ý tưởng:

```text
Text
 ↓
Embedding model
 ↓
Dense vector
```

Ví dụ:

```text
"cat"
→ vector A

"kitten"
→ vector B

"database"
→ vector C
```

Vector A và B có thể gần nhau hơn A và C trong semantic space.

---

## 11.2 Google Embeddings

https://developers.google.com/machine-learning/crash-course/embeddings

Nên học:

- Sparse vs dense representation
- Encoding vs embedding
- Semantic similarity
- Contextual embeddings

---

## 11.3 Similarity

Các khái niệm cần hiểu:

- Dot product
- Cosine similarity
- Euclidean distance

Cosine similarity khái niệm:

```text
cosine_similarity(A, B)
=
(A · B) / (||A|| ||B||)
```

Điểm quan trọng:

> Search theo semantic meaning thay vì chỉ matching keyword.

---

## 11.4 Sentence Transformers

https://www.sbert.net/

Dùng để:

- Sentence embeddings
- Semantic similarity
- Semantic search
- Clustering
- Classification

Trong retrieval pipeline có thể dùng:

```text
Bi-encoder
   ↓
Candidate retrieval
   ↓
Reranker
   ↓
Final ranking
```

---

## 11.5 FAISS

https://faiss.ai/

FAISS là thư viện cho:

- Similarity search
- Clustering
- Dense vectors
- CPU / GPU search

Concept:

```text
Query vector
      ↓
Similarity / Distance
      ↓
Nearest Neighbors
      ↓
Top-k results
```

Cần phân biệt:

```text
Exact nearest-neighbor search
vs
Approximate nearest-neighbor (ANN)
```

Sau đó mới học sâu:

- HNSW
- IVF
- PQ

---

# 12. Phase 9 — RAG Fundamentals

## 12.1 RAG là gì?

Retrieval-Augmented Generation:

```text
User query
    ↓
Retrieve relevant information
    ↓
Relevant chunks
    ↓
Prompt / context augmentation
    ↓
LLM
    ↓
Answer
```

Mục tiêu:

```text
LLM knowledge
+
External / fresh / private knowledge
```

---

## 12.2 ⭐ Original RAG paper

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**

https://arxiv.org/abs/2005.11401

Ý tưởng cốt lõi:

```text
Parametric memory
+
Non-parametric memory
```

---

## 12.3 ⭐ RAG Survey

**Retrieval-Augmented Generation for Large Language Models: A Survey**

https://arxiv.org/abs/2312.10997

Survey giúp hệ thống hóa:

```text
Naive RAG
Advanced RAG
Modular RAG
```

và:

```text
Retrieval
Generation
Augmentation
Evaluation
```

---

# 13. Chunking

Một misunderstanding phổ biến:

> "Chunk size = 500 tokens là chuẩn."

Không có universal chunk size.

Trade-off:

```text
Chunk quá nhỏ
→ thiếu context

Chunk quá lớn
→ retrieval kém
→ embedding bị dilute
→ tốn context window
```

Các chiến lược:

- Fixed-size chunking
- Sentence chunking
- Paragraph chunking
- Recursive chunking
- Structure-aware chunking
- Semantic chunking
- Contextual chunking
- Chunk expansion

---

## Tài liệu chunking

https://www.pinecone.io/learn/chunking-strategies/

---

# 14. Retrieval Pipeline

Một RAG pipeline cơ bản:

```text
Documents
   ↓
Parse
   ↓
Chunk
   ↓
Embedding
   ↓
Vector index
   ↓
User query
   ↓
Query embedding
   ↓
Similarity search
   ↓
Top-k chunks
   ↓
Prompt
   ↓
LLM
   ↓
Answer
```

Sau basic RAG, học:

```text
Reranking
Hybrid Search
Metadata filtering
Query rewriting
Multi-query retrieval
Evaluation
```

---

## Pinecone RAG learning series

https://www.pinecone.io/learn/series/rag/

Có thể dùng để mở rộng từ:

```text
Basic RAG
↓
Advanced retrieval
↓
Reranking
↓
Hybrid search
↓
Evaluation
```

---

# 15. Bộ tài liệu MUST-READ

Nếu phải giảm toàn bộ curriculum xuống 10 resource chính:

## 1. An Introduction to Statistical Learning with Python

**Mục tiêu:** Machine Learning fundamentals

https://www.statlearning.com/home

---

## 2. Google Machine Learning Crash Course

**Mục tiêu:** Intuition + exercises

https://developers.google.com/machine-learning/crash-course

---

## 3. Dive into Deep Learning

**Mục tiêu:** Math + Neural Network + CNN + RNN + Transformer

https://www.d2l.ai/

---

## 4. Karpathy — Neural Networks: Zero to Hero

**Mục tiêu:** Implement neural networks từ đầu

https://github.com/karpathy/nn-zero-to-hero

---

## 5. PyTorch Learn the Basics

**Mục tiêu:** Framework

https://docs.pytorch.org/tutorials/beginner/basics/intro.html

---

## 6. Stanford CS231n

**Mục tiêu:** CNN / Deep Learning / Computer Vision

https://cs231n.stanford.edu/

---

## 7. Stanford CS224N

**Mục tiêu:** NLP / Deep Learning / Transformer / LLM

https://web.stanford.edu/class/cs224n/

---

## 8. Attention Is All You Need

**Mục tiêu:** Transformer fundamentals

https://arxiv.org/abs/1706.03762

---

## 9. Hugging Face LLM Course

**Mục tiêu:** LLM engineering + internals

https://huggingface.co/learn/llm-course/en/chapter1/1

---

## 10. RAG paper + RAG Survey

**Mục tiêu:** Retrieval-Augmented Generation

Original paper:
https://arxiv.org/abs/2005.11401

Survey:
https://arxiv.org/abs/2312.10997

---

# 16. Learning Projects

Không nên chỉ đọc.

Mỗi phase nên có một project.

---

## Project 1 — Linear Regression from Scratch

Chỉ dùng:

```text
Python
NumPy
Matplotlib
```

Tự implement:

```text
y = wx + b
MSE
gradient
gradient descent
```

Phải tự tính được một training step.

---

# Project 2 — Logistic Regression from Scratch

Tự implement:

```text
sigmoid
log loss
gradient descent
classification
confusion matrix
precision
recall
F1
```

Dataset có thể bắt đầu với dataset nhỏ dễ debug.

---

# Project 3 — Neural Network from Scratch

Ở version đầu:

```text
Python + NumPy
```

Tự viết:

```text
Dense
ReLU
Softmax
Cross Entropy
Backpropagation
SGD
```

Không dùng PyTorch cho phần này.

Mục tiêu là hiểu bản chất.

---

# Project 4 — CNN bằng PyTorch

Dataset:

```text
MNIST
hoặc
CIFAR-10
```

Phải trace được:

```text
input shape
→ convolution output
→ activation
→ pooling
→ feature maps
→ flatten
→ classifier
```

---

# Project 5 — LSTM

Có thể làm:

```text
Character-level language model
```

hoặc:

```text
Sentiment classifier
```

Phải hiểu:

```text
hidden state
cell state
forget gate
input gate
output gate
```

---

# Project 6 — Self-Attention From Scratch

Đây là project rất quan trọng.

Không dùng Transformer library.

Tự implement:

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.T
attention = softmax(scores / sqrt(d))
output = attention @ V
```

Sau đó mới đối chiếu với:

```python
torch.nn.MultiheadAttention
```

---

# Project 7 — Tiny GPT

Dùng PyTorch xây mini decoder-only Transformer:

```text
Tokenizer
↓
Embedding
↓
Positional information
↓
Masked Self-Attention
↓
MLP
↓
Residual
↓
LayerNorm
↓
LM Head
↓
Cross Entropy
```

Train trên dataset nhỏ.

Mục tiêu:

> Hiểu GPT trước khi dùng API của LLM.

---

# Project 8 — RAG From Scratch

Version đầu:

```text
No LangChain
No high-level RAG framework
```

Tự build:

```text
Documents
 ↓
Chunk
 ↓
Embedding
 ↓
FAISS
 ↓
Similarity Search
 ↓
Top-k chunks
 ↓
Prompt
 ↓
LLM
```

Sau đó mở rộng:

```text
Reranker
Hybrid search
Metadata filtering
Query rewriting
Multi-query
Evaluation
```

---

# 17. Questions để tự kiểm tra

Nếu chưa trả lời được các câu dưới đây, nên quay lại học phase tương ứng.

## Machine Learning

### Q1
Tại sao logistic regression dùng sigmoid?

### Q2
Tại sao cross-entropy phù hợp cho classification?

### Q3
Precision và recall khác nhau như thế nào?

### Q4
Khi nào nên ưu tiên precision?

### Q5
Khi nào nên ưu tiên recall?

### Q6
Overfitting xảy ra như thế nào?

### Q7
Regularization tác động vào optimization ra sao?

---

## Deep Learning

### Q8
Backpropagation thực sự làm gì?

### Q9
Gradient vanishing xảy ra tại sao?

### Q10
Gradient explosion là gì?

### Q11
SGD khác Adam thế nào?

### Q12
Batch size ảnh hưởng training thế nào?

---

## CNN

### Q13
Tại sao convolution phù hợp cho image?

### Q14
Parameter sharing là gì?

### Q15
Receptive field là gì?

### Q16
Stride và padding ảnh hưởng output shape thế nào?

---

## RNN / LSTM

### Q17
Tại sao RNN khó nhớ long-term dependency?

### Q18
LSTM giải quyết vấn đề gì?

### Q19
Cell state khác hidden state thế nào?

---

## Transformer

### Q20
Tại sao cần Q/K/V?

### Q21
Tại sao dùng `QKᵀ`?

### Q22
Tại sao chia cho `sqrt(d_k)`?

### Q23
Multi-head attention để làm gì?

### Q24
Tại sao cần positional information?

### Q25
Tại sao causal attention phải có mask?

### Q26
Encoder và decoder khác nhau như thế nào?

---

## LLM

### Q27
Token khác word như thế nào?

### Q28
Token ID khác embedding vector thế nào?

### Q29
LLM thực chất dự đoán cái gì?

### Q30
Logits là gì?

### Q31
Softmax làm gì?

### Q32
Temperature tác động vào đâu?

### Q33
Top-k và top-p khác nhau như thế nào?

### Q34
Context window là gì?

### Q35
Tại sao LLM có thể hallucinate?

---

## RAG

### Q36
Tại sao không đưa toàn bộ documents vào prompt?

### Q37
Tại sao cần chunking?

### Q38
Tại sao embedding giúp semantic search?

### Q39
Cosine similarity thực chất đo gì?

### Q40
Vector DB đang tối ưu bài toán gì?

### Q41
Exact search và ANN khác nhau thế nào?

### Q42
Reranker khác retriever thế nào?

### Q43
Tại sao retrieval tốt nhưng answer vẫn có thể sai?

### Q44
Chunking tốt phải cân bằng những gì?

---

# 18. Nguyên tắc học

## 18.1 Đừng học theo kiểu "API-first"

Không nên bắt đầu bằng:

```python
model = SomeLLM(...)
model.generate(...)
```

Mà nên đi:

```text
Math
↓
Algorithm
↓
Architecture
↓
Implementation
↓
Framework
↓
API
```

---

## 18.2 Mỗi concept nên đi qua 3 tầng

### Tầng 1 — Intuition

Ví dụ:

> Attention giúp token tập trung vào các token liên quan.

### Tầng 2 — Math

```text
Attention(Q,K,V)
= softmax(QKᵀ / √d)V
```

### Tầng 3 — Code

```python
scores = Q @ K.T
weights = softmax(scores / sqrt(d))
output = weights @ V
```

Khi hiểu đủ cả 3 tầng, concept mới thực sự chắc.

---

## 18.3 Đọc paper sau khi có intuition

Ví dụ:

```text
Attention intuition
↓
D2L / CS224N
↓
Implement attention
↓
Attention Is All You Need
```

Đừng bắt đầu bằng paper nếu bạn chưa có background.

---

## 18.4 Build before abstraction

Đối với RAG:

```text
Build with NumPy / PyTorch / FAISS
↓
Understand
↓
Then use high-level framework
```

Đối với LLM:

```text
Tiny GPT
↓
Understand Transformer
↓
Hugging Face
↓
Production LLM APIs
```

---

# 19. Roadmap ngắn gọn

Nếu cần một bản cực ngắn để theo mỗi ngày:

```text
1. Math
   ↓
2. Linear Regression
   ↓
3. Logistic Regression
   ↓
4. Precision / Recall / F1
   ↓
5. Overfitting / Regularization
   ↓
6. Neural Network
   ↓
7. Backpropagation
   ↓
8. Gradient Descent
   ↓
9. PyTorch
   ↓
10. CNN
   ↓
11. RNN
   ↓
12. LSTM
   ↓
13. Attention
   ↓
14. Transformer
   ↓
15. Tokenization
   ↓
16. Embedding
   ↓
17. Causal LM
   ↓
18. Logits / Softmax
   ↓
19. Sampling / Temperature
   ↓
20. Context Window
   ↓
21. LLM
   ↓
22. Sentence Embeddings
   ↓
23. Cosine Similarity
   ↓
24. FAISS
   ↓
25. Chunking
   ↓
26. Retrieval
   ↓
27. Reranking
   ↓
28. RAG
   ↓
29. RAG Evaluation
```

---

# 20. Recommended specialization for NLP / Speech / LLM

Nếu mục tiêu nghề nghiệp của bạn thiên về NLP, Speech, LLM, RAG thì không cần đào quá sâu Computer Vision.

Ưu tiên thời gian theo thứ tự:

```text
ML fundamentals
   ↓
Neural Networks
   ↓
PyTorch
   ↓
RNN / LSTM
   ↓
Attention
   ↓
Transformer
   ↓
NLP
   ↓
Tokenizer / Embedding
   ↓
LLM
   ↓
Vector Search
   ↓
RAG
```

CNN vẫn nên học ở mức đủ hiểu kiến trúc và nguyên lý.

---

# 21. Final goal

Sau khi hoàn thành roadmap, mục tiêu không phải là:

> "Biết dùng OpenAI API / Hugging Face / LangChain."

Mục tiêu là có thể nhìn một hệ thống AI và giải thích được:

```text
Input
 ↓
Preprocessing
 ↓
Representation / Embedding
 ↓
Model Architecture
 ↓
Forward Pass
 ↓
Loss
 ↓
Optimization
 ↓
Inference
 ↓
Decoding
 ↓
Retrieval (nếu có)
 ↓
Generation
```

và khi gặp một vấn đề mới có thể tự hỏi:

```text
Problem nằm ở:
- data?
- representation?
- model architecture?
- loss?
- optimization?
- inference?
- retrieval?
- context?
- decoding?
- evaluation?
```

Đó là nền tảng để chuyển từ việc **"biết sử dụng AI tools"** sang **"hiểu và xây AI systems"**.
