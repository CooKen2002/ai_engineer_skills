### Tầng 1 — Core AI/ML Knowledge (must-have, xuất hiện ở gần 100% JD)

- **Machine Learning cơ bản**: Regression, Classification, đánh giá mô hình (precision/recall/F1)
- **Deep Learning**: CNN, RNN, Transformer — hiểu kiến trúc, không chỉ biết gọi API
- **Ít nhất 1 framework**: PyTorch hoặc TensorFlow (PyTorch đang là lựa chọn phổ biến hơn ở hầu hết JD)
- **LLM fundamentals**: prompt, context window, token, temperature, cách LLM sinh văn bản
- **RAG fundamentals**: chunking, embedding, vector search — hiểu bản chất tại sao cần, không chỉ biết dùng thư viện


1. Machine Learning cơ bản
A. ⭐ Must-read: An Introduction to Statistical Learning with Python

Đây là cuốn mình khuyên bạn bắt đầu đầu tiên.

Nó bao phủ đúng những phần bạn cần:

Regression
Classification
Resampling
Regularization
Tree-based models
SVM
Deep Learning
Unsupervised Learning

Đặc biệt có Python labs ở cuối mỗi chapter, nên không chỉ đọc lý thuyết. Website chính thức cung cấp PDF và notebook.

ISLP — An Introduction to Statistical Learning with Python

Nên đọc các chapter nào?

Mình ưu tiên:

Topic	Priority
Introduction	⭐⭐⭐
Linear Regression	⭐⭐⭐⭐⭐
Classification	⭐⭐⭐⭐⭐
Resampling	⭐⭐⭐⭐⭐
Regularization	⭐⭐⭐⭐⭐
Tree-Based Methods	⭐⭐⭐
SVM	⭐⭐⭐
Deep Learning	⭐⭐⭐⭐⭐
Unsupervised Learning	⭐⭐⭐
B. ⭐ Google Machine Learning Crash Course

Nếu ISLP hơi nặng về sách, học song song Google MLCC sẽ dễ vào hơn.

Google hiện có các module riêng cho:

Linear Regression
Logistic Regression
Classification
Overfitting / Generalization
Neural Networks
Embeddings
Intro to LLMs
Production ML

và có interactive exercises chạy trực tiếp trên Colab.

Google Machine Learning Crash Course

Đặc biệt đọc phần này

Precision / Recall / F1 / threshold / ROC-AUC

Google giải thích rất rõ TP, TN, FP, FN và trade-off giữa precision với recall.

Classification metrics — Google

Điều cần hiểu không phải chỉ là:

precision = TP / (TP + FP)
recall    = TP / (TP + FN)

mà phải trả lời được:

"Trong bài toán thực tế, tại sao tôi ưu tiên precision hơn recall?"

Ví dụ với speech/NLP:

False Positive = hệ thống nhận nhầm intent
False Negative = hệ thống bỏ sót intent

Đó mới là tư duy ML thực sự.

2. Mathematical foundation

Bạn không cần trở thành mathematician, nhưng phải đủ hiểu:

Linear Algebra
Vector
Matrix
Dot product
Matrix multiplication
Norm
Eigenvalue/eigenvector
Projection
Calculus
Derivative
Partial derivative
Gradient
Chain rule
Probability
Probability distribution
Expectation
Variance
Conditional probability
Bayes theorem
Maximum likelihood
Optimization
Loss function
Gradient descent
Learning rate
Local/global minima
Regularization

Một resource rất tốt là phần Preliminaries của Dive into Deep Learning, vì nó kết hợp math + code + visualization.

3. Deep Learning
⭐ Must-read: Dive into Deep Learning

Đây có lẽ là resource tốt nhất để đi từ:

math → neural network → CNN → RNN → Transformer

D2L là một interactive book, mỗi section có:

lý thuyết
mathematical derivation
figures
executable Jupyter notebook
PyTorch implementation

và có nhiều implementation framework, trong đó có PyTorch.

Dive into Deep Learning

Với bạn, nên học:
2. Preliminaries
3. Linear Neural Networks
4. Multilayer Perceptrons
5. Deep Learning Computation
6. Convolutional Neural Networks
9. Modern Convolutional Neural Networks
10. Recurrent Neural Networks
11. Modern Recurrent Neural Networks
12. Attention Mechanisms and Transformers
4. ⭐ Neural Network từ đầu — cực kỳ đáng học
Andrej Karpathy — Neural Networks: Zero to Hero

Mình đánh giá resource này rất cao cho developer.

Thay vì:

model = SomeLLM(...)
model.generate(...)

Karpathy bắt bạn tự xây:

neuron
    ↓
forward
    ↓
loss
    ↓
backpropagation
    ↓
gradient descent
    ↓
neural network
    ↓
language model
    ↓
Transformer / GPT

Course bắt đầu từ neural network và backpropagation, sau đó đi tới language modeling và Transformer. Code/notebook của từng lecture đều được publish.

Neural Networks: Zero to Hero

Bài cực quan trọng

micrograd

Bạn sẽ hiểu thực sự:

y = f(x)
      ↓
loss
      ↓
∂loss/∂x
      ↓
gradient
      ↓
update weights

Sau đó mới chuyển sang PyTorch autograd.

Đây là một trong những cách tốt nhất để thoát khỏi kiểu:

"PyTorch tự làm backprop nên tôi không cần biết nó hoạt động thế nào."

5. Deep Learning textbook — dùng như reference
Deep Learning — Goodfellow, Bengio, Courville

Đây là textbook kinh điển về Deep Learning và toàn bộ nội dung online được cung cấp miễn phí.

Deep Learning — Goodfellow, Bengio, Courville

Mình không khuyên đọc cover-to-cover ngay từ đầu.

Dùng nó khi bạn gặp:

Backpropagation
Optimization
Regularization
Representation Learning
CNN
Sequence Models

và muốn đào sâu hơn.

6. PyTorch — framework bạn nên chọn

Với mục tiêu của bạn, mình chọn:

PyTorch

Không cần học cả PyTorch và TensorFlow lúc này.

Official PyTorch tutorial hiện có một learning path rất rõ:

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

và có notebook chạy trên Colab.

PyTorch — Learn the Basics

Nhưng đừng chỉ học API

Bạn phải hiểu được đoạn này:

loss.backward()
optimizer.step()
optimizer.zero_grad()

thực chất đang làm gì.

Phải giải thích được:

forward pass
     ↓
compute loss
     ↓
backpropagation
     ↓
compute gradients
     ↓
optimizer updates parameters
7. CNN
⭐ Stanford CS231n

CS231n là resource kinh điển để hiểu CNN dưới góc nhìn kiến trúc.

Course tập trung vào việc hiểu và tự implement/train neural networks cho computer vision, thay vì chỉ gọi model có sẵn.

Stanford CS231n

Bạn cần hiểu:

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

và quan trọng hơn:

Tại sao CNN cần convolution?
Tại sao weight sharing giúp giảm parameters?
receptive field là gì?
stride/padding ảnh hưởng output thế nào?
Paper nên đọc
ResNet

Deep Residual Learning for Image Recognition

Paper này giúp hiểu tại sao mạng rất sâu khó train và tại sao residual connection lại quan trọng.

ResNet paper — arXiv

Không cần học thuộc paper. Chỉ cần hiểu:

x → layers → F(x)

vs

x → F(x) + x
8. RNN / LSTM

RNN rất đáng học dù LLM hiện tại chủ yếu dùng Transformer.

Lý do:

Bạn sẽ hiểu Transformer xuất hiện để giải quyết vấn đề gì.

Concept:

x1 → RNN → h1
x2 → RNN → h2
x3 → RNN → h3
...

và:

h_t = f(x_t, h_(t-1))

Sau đó hiểu:

vanishing gradient
exploding gradient
long-term dependency
hidden state
sequence modeling
⭐ Paper: LSTM

Long Short-Term Memory — Hochreiter & Schmidhuber

Đây là paper gốc của LSTM, xuất bản năm 1997.

Long Short-Term Memory — original paper

Sau khi hiểu LSTM, bạn sẽ nhìn thấy lý do Transformer hấp dẫn hơn rất nhiều.

9. Transformer

Đây là phần quan trọng nhất nếu mục tiêu cuối cùng là LLM.

⭐⭐⭐ Attention Is All You Need

Paper gốc của Transformer.

Attention Is All You Need — arXiv

Paper đề xuất kiến trúc Transformer dựa hoàn toàn trên attention, không cần recurrence hoặc convolution như sequence models trước đó.

Bạn phải hiểu được:

Input tokens
      ↓
Embedding
      ↓
Positional information
      ↓
Self Attention
      ↓
Multi-Head Attention
      ↓
Feed Forward Network
      ↓
Residual + LayerNorm
      ↓
Output

và đặc biệt:

Self-attention

Phải tự tính được:

Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
    = softmax(QKᵀ / √d)V

Nếu bạn hiểu được công thức này thì kiến trúc Transformer sẽ bớt "magic" rất nhiều.

10. Stanford CS224N — cực kỳ phù hợp với mục tiêu LLM

CS224N hiện vẫn là một trong những course rất tốt để học NLP + Deep Learning. Course Winter 2026 bao phủ từ nền tảng NLP đến các chủ đề LLM hiện đại và dùng PyTorch.

Stanford CS224N — NLP with Deep Learning

Đây là course mình đặc biệt khuyên bạn học sau CS231n/D2L.

Vì nó nối:

NLP
 ↓
Word embeddings
 ↓
RNN
 ↓
Attention
 ↓
Transformer
 ↓
LLM
11. LLM fundamentals

Bây giờ mới bước vào LLM.

⭐ Hugging Face LLM Course

Đây là resource rất tốt cho:

Transformers
tokenizer
pretrained models
fine-tuning
causal language modeling
datasets
inference
advanced LLM topics

Course hiện có tới Chapter 12 và có nội dung về fine-tuning, dataset curation và reasoning models.

Hugging Face LLM Course

12. Token — phải hiểu thật kỹ

Bạn cần hiểu:

text
 ↓
tokenizer
 ↓
token IDs
 ↓
embedding vectors
 ↓
Transformer
 ↓
logits
 ↓
probabilities
 ↓
sample next token
 ↓
repeat

Tokenizer thực tế không đơn giản chỉ là:

text.split(" ")

Hugging Face Tokenizers mô tả pipeline gồm:

Normalization
↓
Pre-tokenization
↓
Tokenization model
↓
Post-processing

và các tokenizer hiện đại thường sử dụng subword techniques như BPE.

Hugging Face Tokenizers documentation

13. "LLM sinh text như thế nào?"

Đây là concept bạn bắt buộc phải tự giải thích được.

Ví dụ:

"The cat is"

model không "nghĩ" nguyên câu.

Nó dự đoán phân phối:

sat       0.42
sleeping  0.16
big       0.08
...

sau đó decoding strategy chọn token tiếp theo.

Lặp lại:

"The cat is"
        ↓
"The cat is sitting"
        ↓
"The cat is sitting on"
        ↓
"The cat is sitting on the"
        ↓
...

Hugging Face hiện có tài liệu riêng về decoding strategies như greedy search, sampling, top-k, top-p...

Hugging Face — Generation Strategies

14. Temperature

Bạn cần hiểu temperature ở mức toán học chứ không phải:

"temperature thấp = deterministic"

Mà phải hiểu nó tác động lên logits trước softmax.

Về concept:

logits
   ↓
divide by temperature
   ↓
softmax
   ↓
probability distribution
   ↓
sampling

Từ đó mới hiểu tại sao temperature cao làm distribution "phẳng" hơn và temperature thấp làm distribution tập trung hơn.

15. GPT-2 paper — cực đáng đọc

Sau Transformer, đọc:

Language Models are Unsupervised Multitask Learners

Paper GPT-2 rất hữu ích để nối Transformer với modern language model.

Paper mô tả GPT-2 như một Transformer language model được train trên lượng lớn text và sử dụng autoregressive language modeling để dự đoán chuỗi.

Điểm hay nữa là paper có giải thích representation/tokenization và BPE.

GPT-2 paper PDF

16. RAG — bắt đầu từ Embedding

Trước RAG hãy chắc chắn bạn hiểu:

text
 ↓
embedding model
 ↓
vector

"cat"      → [0.21, -0.31, ...]
"kitten"   → [0.22, -0.29, ...]
"database" → [-0.52, 0.81, ...]

Ý tưởng là biểu diễn semantic information thành vector để có thể so sánh similarity.

Google có một module khá tốt về embeddings, bao gồm:

sparse vs dense representation
encoding vs embedding
semantic similarity
contextual embeddings

Google — Embeddings

17. Sentence Transformers

Sau đó đọc Sentence Transformers để hiểu cách embedding hiện đại được sử dụng trong semantic search.

Nó tạo fixed-size vector representations và hỗ trợ semantic similarity, semantic search, clustering, classification; trong retrieval pipelines thường có thể dùng bi-encoder trước rồi reranker sau.

Sentence Transformers documentation

18. Vector Search

Đây là chỗ rất nhiều developer chỉ biết:

vector_db.similarity_search(...)

nhưng không biết bên dưới làm gì.

Một resource tốt là FAISS.

FAISS là thư viện cho similarity search và clustering trên dense vectors, với các thuật toán chạy CPU/GPU và các cơ chế index để tìm nearest vectors hiệu quả.

FAISS documentation

Bạn cần hiểu:

Query vector
      ↓
similarity / distance
      ↓
nearest neighbors
      ↓
top-k results

và sau đó mới học:

Exact search
vs
Approximate Nearest Neighbor

rồi mới đi vào HNSW / IVF / PQ.

19. ⭐ RAG paper

Đọc paper gốc:

Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Paper này formalize ý tưởng kết hợp:

parametric memory
+
non-parametric memory

tức model knowledge + external retrieved knowledge.

Original RAG paper — arXiv

20. RAG Survey

Sau paper gốc, đọc:

Retrieval-Augmented Generation for Large Language Models: A Survey

Survey phân chia và phân tích:

Naive RAG
Advanced RAG
Modular RAG

và các thành phần:

Retrieval
Generation
Augmentation
Evaluation

RAG Survey — arXiv

21. Chunking — đừng học kiểu "500 tokens là chuẩn"

Đây là một misunderstanding rất phổ biến.

Chunking thực chất là bài toán:

Too small
→ mất context

Too large
→ embedding bị dilute
→ retrieval kém
→ tốn context

Không có một chunk size universal.

Một hướng dẫn hiện tại về chunking phân tích:

fixed-size chunking
sentence/paragraph chunking
recursive chunking
structure-aware chunking
semantic chunking
contextual chunking
chunk expansion

Pinecone — Chunking Strategies for LLM Applications

22. Một tài liệu RAG thực hành tốt

Pinecone có một chuỗi tài liệu RAG khá tốt để chuyển từ:

basic RAG

sang:

two-stage retrieval
reranking
hybrid search
multi-query
metadata
evaluation

Pinecone — RAG learning series

23. Bộ "must-read" mình thực sự khuyên bạn

Nếu phải giảm toàn bộ danh sách trên xuống 10 resource, mình chọn:

1.

An Introduction to Statistical Learning with Python

→ ML fundamentals

2.

Google Machine Learning Crash Course

→ intuition + exercises

3.

Dive into Deep Learning

→ math + neural network + CNN + RNN + Transformer

4.

Karpathy — Neural Networks: Zero to Hero

→ implement neural networks từ đầu

5.

PyTorch Learn the Basics

→ framework

6.

Stanford CS231n

→ CNN/deep learning architecture

7.

Stanford CS224N

→ NLP / Transformer / LLM

8.

Attention Is All You Need

→ Transformer paper

9.

Hugging Face LLM Course

→ LLM engineering + internals

10.

RAG paper + RAG Survey

→ RAG fundamentals

24. Learning path mình đề xuất cho bạn

Mình sẽ không học theo thứ tự "ML → đọc 20 paper → mới code".

Nên học kiểu:

PHASE 0
Math fundamentals
        ↓
PHASE 1
Linear Regression
Logistic Regression
Classification
Precision / Recall / F1
Overfitting
Regularization
        ↓
PHASE 2
Neural Network
Forward
Loss
Backprop
Gradient Descent
        ↓
PHASE 3
PyTorch
Tensor
Dataset
Model
Autograd
Optimizer
Training Loop
        ↓
PHASE 4
CNN
        ↓
RNN
LSTM
        ↓
Attention
        ↓
Transformer
        ↓
PHASE 5
Tokenizer
Embedding
Causal LM
Logits
Softmax
Sampling
Temperature
Context Window
        ↓
PHASE 6
LLM
        ↓
PHASE 7
Embedding
Vector Similarity
ANN Search
        ↓
PHASE 8
RAG
Chunking
Embedding
Retrieval
Reranking
Generation
Evaluation
25. Quan trọng nhất: học bằng project

Mình khuyên bạn không đọc hết sách trước.

Hãy tạo một chuỗi mini-project:

Project 1 — Linear Regression from scratch

Chỉ dùng:

Python
NumPy
Matplotlib

Tự implement:

y = wx + b
MSE
gradient
gradient descent
Project 2 — Logistic Regression

Tự implement:

sigmoid
log loss
gradient descent
classification
precision
recall
F1
confusion matrix
Project 3 — Neural Network from scratch

Không PyTorch.

Tự viết:

Dense
ReLU
Softmax
Cross Entropy
Backpropagation
SGD

Sau đó mới chuyển sang PyTorch.

Project 4 — CNN bằng PyTorch

Train:

MNIST / CIFAR-10

và phải tự trả lời được:

input shape
→ conv output shape
→ pooling
→ feature maps
→ flatten
→ classifier
Project 5 — LSTM

Làm một:

character-level language model

hoặc sentiment classifier.

Mục tiêu là hiểu:

hidden state
cell state
forget gate
input gate
output gate
Project 6 — Self-Attention from scratch

Đây là project mình đặc biệt khuyên.

Không dùng Transformer library.

Tự implement:

Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.T
attention = softmax(scores / sqrt(d))
output = attention @ V

Sau đó mới dùng PyTorch nn.MultiheadAttention hoặc Transformer.

Project 7 — Tiny GPT

Dùng PyTorch build một mini decoder-only Transformer:

Tokenizer
↓
Embedding
↓
Positional encoding
↓
Masked self-attention
↓
MLP
↓
Residual
↓
LayerNorm
↓
LM Head
↓
Cross entropy

Train trên một dataset nhỏ.

Sau project này, rất nhiều khái niệm LLM sẽ tự nhiên trở nên rõ ràng.

Project 8 — RAG from scratch

Đừng dùng LangChain ở version đầu.

Tự build:

PDF/text
 ↓
chunk
 ↓
embedding
 ↓
FAISS
 ↓
similarity search
 ↓
top-k chunks
 ↓
prompt
 ↓
LLM

Sau đó mới thêm:

reranker
hybrid search
metadata filtering
query rewriting
evaluation
26. Khi học xong, bạn phải tự trả lời được những câu này

Nếu chưa trả lời được, nghĩa là chưa thực sự nắm core knowledge.

ML

Tại sao logistic regression dùng sigmoid?

Tại sao cross-entropy phù hợp cho classification?

Precision và recall trade-off vì đâu?

Overfitting xảy ra như thế nào?

Regularization tác động vào optimization ra sao?

Deep Learning

Backpropagation thực sự làm gì?

Tại sao gradient vanishing xảy ra?

Adam khác SGD như thế nào?

Batch size ảnh hưởng training thế nào?

CNN

Tại sao convolution phù hợp cho image?

Parameter sharing là gì?

Receptive field là gì?

RNN

Tại sao RNN khó nhớ long-term dependency?

LSTM giải quyết vấn đề gì?

Transformer

Tại sao cần Q/K/V?

Tại sao có QKᵀ?

Tại sao chia cho sqrt(d_k)?

Tại sao cần multi-head?

Tại sao cần positional information?

Tại sao causal attention phải có mask?

LLM

Token khác word như thế nào?

Embedding khác token ID như thế nào?

LLM thực chất dự đoán cái gì?

Logits là gì?

Softmax làm gì?

Temperature tác động vào đâu?

Context window là gì?

Tại sao model có thể hallucinate?

RAG

Tại sao không đưa toàn bộ documents vào prompt?

Tại sao cần chunking?

Tại sao embedding giúp semantic search?

Cosine similarity thực chất đo gì?

Vector DB đang tối ưu bài toán gì?

Tại sao retrieval tốt nhưng câu trả lời vẫn sai?

Reranker khác retriever thế nào?