# Core AI/ML — Tầng 3: Chuyên sâu theo Domain

> **Mục tiêu của Tầng 3:** sau khi đã hoàn thành Tầng 1 (Core AI/ML) và Tầng 2 (Production AI Engineering), đi sâu vào một hoặc hai domain đủ để **tự thiết kế, huấn luyện/fine-tune, đánh giá và triển khai hệ thống AI chuyên biệt**.
>
> Với định hướng hiện tại, thứ tự khuyến nghị là:
>
> **Speech/NLP = Core specialization**
>
> **Computer Vision & Multimodal = Secondary specialization**
>
> Recommendation Systems chỉ học nếu JD / sản phẩm thực sự yêu cầu.

---

## Mục lục

1. [Tầng 3 là gì?](#1-tầng-3-là-gì)
2. [Nguyên tắc chọn specialization](#2-nguyên-tắc-chọn-specialization)
3. [Roadmap tổng thể](#3-roadmap-tổng-thể)
4. [Track A — Speech/NLP](#4-track-a--speechnlp)
5. [Speech foundations](#5-speech-foundations)
6. [ASR / STT](#6-asr--stt)
7. [ASR architecture](#7-asr-architecture)
8. [Whisper](#8-whisper)
9. [CTranslate2 / faster-whisper](#9-ctranslate2--faster-whisper)
10. [Wav2Vec 2.0](#10-wav2vec-20)
11. [Conformer / CTC / RNN-T](#11-conformer--ctc--rnn-t)
12. [VAD / timestamps / forced alignment](#12-vad--timestamps--forced-alignment)
13. [Speaker diarization](#13-speaker-diarization)
14. [Speech enhancement](#14-speech-enhancement)
15. [ASR evaluation](#15-asr-evaluation)
16. [TTS](#16-tts)
17. [TTS architecture](#17-tts-architecture)
18. [TTS evaluation](#18-tts-evaluation)
19. [NLP specialization](#19-nlp-specialization)
20. [Dialogue systems](#20-dialogue-systems)
21. [Rasa / dialogue management](#21-rasa--dialogue-management)
22. [LLM fine-tuning](#22-llm-fine-tuning)
23. [SFT](#23-sft)
24. [PEFT / LoRA](#24-peft--lora)
25. [QLoRA](#25-qlora)
26. [Preference tuning / DPO](#26-preference-tuning--dpo)
27. [Fine-tuning data engineering](#27-fine-tuning-data-engineering)
28. [Fine-tuning evaluation](#28-fine-tuning-evaluation)
29. [Khi nào RAG tốt hơn fine-tuning?](#29-khi-nào-rag-tốt-hơn-fine-tuning)
30. [Track B — Computer Vision](#30-track-b--computer-vision)
31. [CV foundations](#31-cv-foundations)
32. [Image segmentation](#32-image-segmentation)
33. [Segmentation architectures](#33-segmentation-architectures)
34. [Segmentation evaluation](#34-segmentation-evaluation)
35. [SAM / SAM 2](#35-sam--sam-2)
36. [Track C — Vision-Language / Multimodal](#36-track-c--vision-language--multimodal)
37. [VLM architecture](#37-vlm-architecture)
38. [VLM inference](#38-vlm-inference)
39. [VLM evaluation](#39-vlm-evaluation)
40. [CV + LLM integration](#40-cv--llm-integration)
41. [Document AI / OCR + LLM](#41-document-ai--ocr--llm)
42. [Audio + LLM / speech-language models](#42-audio--llm--speech-language-models)
43. [Optional Track — Recommendation Systems](#43-optional-track--recommendation-systems)
44. [Hands-on Project 1 — Production Speech Pipeline](#44-hands-on-project-1--production-speech-pipeline)
45. [Hands-on Project 2 — Fine-tuned Domain LLM](#45-hands-on-project-2--fine-tuned-domain-llm)
46. [Hands-on Project 3 — Dialogue / Voice Assistant](#46-hands-on-project-3--dialogue--voice-assistant)
47. [Hands-on Project 4 — Segmentation + VLM](#47-hands-on-project-4--segmentation--vlm)
48. [Capstone — Multimodal Voice AI Assistant](#48-capstone--multimodal-voice-ai-assistant)
49. [Interview checklist](#49-interview-checklist)
50. [Bộ tài liệu ưu tiên](#50-bộ-tài-liệu-ưu-tiên)
51. [Skill matrix](#51-skill-matrix)
52. [Definition of Done](#52-definition-of-done)
53. [Roadmap học theo thứ tự](#53-roadmap-học-theo-thứ-tự)

---

# 1. Tầng 3 là gì?

Tầng 1 trả lời:

> **Model hoạt động như thế nào?**

Tầng 2 trả lời:

> **Làm sao biến model thành một sản phẩm AI chạy ổn định, có thể monitor, scale, đánh giá và maintain?**

Tầng 3 trả lời:

> **Làm sao trở thành người có chiều sâu domain, biết chọn data, architecture, adaptation method, metric và production strategy cho một bài toán cụ thể?**

Một AI Engineer domain-specialized cần nối được:

```text
Domain problem
    ↓
Data characteristics
    ↓
Model family
    ↓
Training / adaptation strategy
    ↓
Evaluation metric
    ↓
Failure modes
    ↓
Inference constraints
    ↓
Production integration
```

---

# 2. Nguyên tắc chọn specialization

Không nên học:

```text
Speech + NLP + CV + Recommendation + Robotics
```

tất cả ở cùng độ sâu.

Nên chọn:

```text
Primary domain
    ↓
Deep specialization
    ↓
Secondary domain
    ↓
Working knowledge
```

## Khuyến nghị

### PRIMARY
```text
Speech + NLP + Dialogue + LLM
```

### SECONDARY
```text
Computer Vision + Multimodal
```

### OPTIONAL
```text
Recommendation Systems
```

---

# 3. Roadmap tổng thể

```text
                    TẦNG 1
        Core ML / DL / Transformer / LLM
                       │
                       ▼
                    TẦNG 2
       RAG / Agents / MCP / API / Serving
                       │
                       ▼
                 ┌─────────────┐
                 │   TẦNG 3    │
                 └──────┬──────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   SPEECH / NLP                CV / MULTIMODAL
          │                           │
    Speech signal                Vision basics
          │                           │
         ASR                      Segmentation
          │                           │
      Whisper                    Detection
          │                           │
    Wav2Vec2 / SSL                  SAM 2
          │                           │
  CTC / RNN-T / Conformer            VLM
          │                           │
 Diarization / Alignment      Vision + LLM
          │                           │
         TTS                     Document AI
          │                           │
       Dialogue                  Multimodal
          │
   LLM fine-tuning
          │
          ▼
     Voice AI system
```

---

# 4. Track A — Speech/NLP

Đây là track nên học sâu nhất.

```text
Audio
 ↓
Signal processing
 ↓
VAD / preprocessing
 ↓
ASR
 ↓
Text normalization
 ↓
NLP / NLU
 ↓
Dialogue / Agent
 ↓
LLM
 ↓
TTS
 ↓
Audio
```

Các bài toán mở rộng:

```text
Audio
 ├── speaker diarization
 ├── language identification
 ├── emotion / paralinguistics
 ├── speech enhancement
 └── voice activity detection
```

---

# 5. Speech foundations

## 5.1 Waveform

Audio số hóa:

```text
continuous sound
       ↓
sampling
       ↓
discrete waveform
```

Cần biết:

- sample rate
- bit depth
- mono / stereo
- amplitude
- dynamic range

Ví dụ:

```text
16 kHz
→ 16,000 samples / second
```

## 5.2 Spectrogram

```text
waveform
   ↓
STFT
   ↓
spectrogram
```

Cần biết:

- FFT
- STFT
- window
- hop length
- frequency bins
- magnitude
- log spectrogram

Mel spectrogram:

```text
waveform
 ↓
STFT
 ↓
Mel filter bank
 ↓
log
 ↓
mel spectrogram
```

## 5.3 MFCC

MFCC vẫn đáng biết vì có giá trị lịch sử và tốt cho việc hiểu feature engineering cổ điển, nhưng không cần ưu tiên nếu pipeline dùng end-to-end Transformer speech models.

## 5.4 Speech preprocessing

```text
raw audio
 ↓
resample
 ↓
channel normalize
 ↓
silence / VAD
 ↓
noise handling
 ↓
normalization
 ↓
model
```

Không nên enhancement quá mạnh làm mất phonetic information.

---

# 6. ASR / STT

ASR:

```text
audio → text
```

Các biến thể thực tế:

- multilingual ASR
- code-switching
- streaming ASR
- offline ASR
- domain vocabulary
- timestamped transcription
- speaker-attributed transcription

---

# 7. ASR architecture

## 7.1 CTC

```text
Audio
 ↓
Encoder
 ↓
frame-level probabilities
 ↓
CTC decoding
 ↓
Text
```

Ưu:

- concept đơn giản
- alignment tự nhiên
- phù hợp nhiều pipeline encoder-only

Nhược:

- conditional independence assumptions
- decoding / language modeling có thể cần thêm components

## 7.2 Encoder-Decoder

```text
Audio
 ↓
Encoder
 ↓
latent representation
 ↓
Autoregressive Decoder
 ↓
Text
```

Whisper thuộc nhóm sequence-to-sequence Transformer.

## 7.3 RNN-T / Transducer

Phù hợp cho streaming:

```text
Audio encoder
      +
Prediction network
      ↓
Joint network
      ↓
Tokens
```

Cần biết:

- encoder
- prediction network
- joint network
- streaming constraint
- emission latency

## 7.4 So sánh

| Architecture | Điểm mạnh | Use case |
|---|---|---|
| CTC | đơn giản, alignment tốt | offline / fine-tuning |
| Seq2Seq | flexible, multilingual, multitask | Whisper-style ASR |
| RNN-T | streaming | realtime assistant |
| Hybrid | nhiều decoder / training signal | production speech |

---

# 8. Whisper

Official repository:

https://github.com/openai/whisper

Whisper là Transformer sequence-to-sequence model cho multilingual speech recognition, speech translation, language identification và các speech tasks liên quan.

Mental model:

```text
Audio
 ↓
log-Mel spectrogram
 ↓
Transformer Encoder
 ↓
Decoder
 ↓
tokens
 ↓
text
```

Cần hiểu:

- audio preprocessing
- log-Mel spectrogram
- encoder
- decoder
- special tokens
- language token
- task token
- autoregressive decoding
- timestamp tokens
- hallucination failure modes

Whisper repository mô tả inference theo các cửa sổ audio 30 giây trượt.

---

# 9. CTranslate2 / faster-whisper

Một stack rất đáng biết cho deployment STT:

```text
Whisper
 ↓
CTranslate2
 ↓
optimized inference
```

CTranslate2 là C++/Python library cho efficient inference của Transformer models.

Official CTranslate2:

https://github.com/OpenNMT/CTranslate2

faster-whisper:

https://github.com/SYSTRAN/faster-whisper

Mental model:

```text
Original model
     ↓
optimized representation
     ↓
CTranslate2 runtime
     ↓
CPU / GPU
```

Các concept cần học:

- model conversion
- quantized compute types
- batching
- beam search
- CPU threading
- GPU inference
- memory layout

Đây là một nhánh rất phù hợp nếu mục tiêu là:

```text
Windows
DLL
offline STT
low latency
```

---

# 10. Wav2Vec 2.0

Paper:

https://arxiv.org/abs/2006.11477

Wav2Vec 2.0 giúp hiểu self-supervised learning cho speech:

```text
raw audio
 ↓
latent speech representation
 ↓
mask
 ↓
contrastive objective
 ↓
pretrained speech encoder
 ↓
fine-tuning with transcripts
```

Phải hiểu:

- unlabeled pretraining
- masked latent representations
- quantization / contrastive target
- fine-tuning with limited labels

---

# 11. Conformer / CTC / RNN-T

## Conformer

Conformer kết hợp:

```text
Convolution
+
Self-Attention
```

để capture:

```text
local patterns
+
global dependencies
```

Paper:

https://arxiv.org/abs/2005.08100

## NVIDIA NeMo

NeMo hiện hỗ trợ ASR, speaker diarization, TTS và speech-language-model workflows.

Official:

https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html

ASR docs:

https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html

Tutorials:

https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html

Nên học sau khi đã hiểu:

```text
ASR fundamentals
↓
Whisper / Wav2Vec2
↓
CTC / RNN-T
↓
Conformer
```

NeMo sau đó dùng để học:

```text
training
fine-tuning
streaming
multilingual ASR
TTS
diarization
```

---

# 12. VAD / timestamps / forced alignment

## VAD

```text
audio
 ↓
speech / non-speech
```

Dùng để:

- giảm compute
- split utterances
- streaming
- remove silence

## Segment timestamps

```json
{
  "start": 2.13,
  "end": 5.91,
  "text": "xin chào"
}
```

## Word-level timestamps

Có thể lưu:

```text
word
start
end
confidence
```

## Forced alignment

Bài toán:

```text
audio + known transcript
        ↓
word / phoneme timestamps
```

Concept:

```text
audio
 ↓
frame probabilities
 ↓
trellis
 ↓
best path
 ↓
aligned timestamps
```

PyTorch/TorchAudio có forced alignment tutorials, nhưng TorchAudio hiện ở maintenance phase và audio/video I/O đã được chuyển dần sang TorchCodec. Hãy kiểm tra version API trước khi dùng tutorial cũ.

Official:

https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html

---

# 13. Speaker diarization

Problem:

> Who spoke when?

Output:

```text
00:00 - 00:04 Speaker A
00:04 - 00:07 Speaker B
00:07 - 00:10 Speaker A
```

Pipeline:

```text
audio
 ↓
VAD / segmentation
 ↓
speaker embeddings
 ↓
clustering / neural diarization
 ↓
speaker labels
```

pyannote.audio:

https://github.com/pyannote/pyannote-audio

NeMo diarization:

https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html

Cần hiểu:

- speaker embedding
- overlap speech
- number of speakers
- clustering
- diarization error rate (DER)

---

# 14. Speech enhancement

Các vấn đề:

```text
noise
reverb
echo
background music
multiple speakers
far-field speech
```

Concept:

- denoising
- dereverberation
- echo cancellation
- SNR
- clipping
- gain normalization

Nguyên tắc:

```text
Measure first
→ enhance only when needed
```

---

# 15. ASR evaluation

## WER

```text
WER = (S + D + I) / N
```

Trong đó:

```text
S = substitutions
D = deletions
I = insertions
N = reference words
```

## CER

```text
CER = character edit errors / reference characters
```

Đặc biệt có ích trong một số ngôn ngữ / bài toán normalization khác nhau.

## Normalization

Trước khi tính metric phải thống nhất:

```text
lowercase?
punctuation?
numbers?
unicode?
diacritics?
spoken numbers?
```

## Domain-specific metrics

Không nên chỉ đo WER.

Với booking assistant nên thêm:

```text
Intent Accuracy
Entity F1
Slot Accuracy
Task Success Rate
```

Ví dụ:

```text
ASR:
"47 Trần Khát Chân"
→ "47 Trần Khát Chấn"
```

WER có thể thấp nhưng address entity đã sai.

---

# 16. TTS

```text
Text → Speech
```

Production TTS gồm:

- text normalization
- pronunciation
- prosody
- speaker identity
- multilingual
- style control
- voice cloning
- latency
- streaming

---

# 17. TTS architecture

Architecture truyền thống:

```text
Text
 ↓
Text normalization
 ↓
Phoneme / linguistic representation
 ↓
Acoustic model
 ↓
Mel spectrogram
 ↓
Vocoder
 ↓
Waveform
```

Các family đáng biết:

```text
Tacotron-style
FastSpeech-style
VITS-style
codec / token based
```

Không cần học tất cả.

Phải hiểu:

```text
text representation
→ acoustic representation
→ waveform synthesis
```

NeMo TTS:

https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/configs.html

Tutorial index:

https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html

---

# 18. TTS evaluation

Không có một metric duy nhất.

## Naturalness

```text
MOS
```

## Speaker similarity

```text
generated voice
vs
target speaker
```

## Intelligibility

Có thể dùng:

```text
text
 ↓
TTS
 ↓
ASR
 ↓
WER
```

## Production

```text
TTFB
latency
streaming latency
artifact rate
failure rate
```

---

# 19. NLP specialization

NLP không chỉ là LLM.

Nên hiểu:

```text
Text classification
NER
Token classification
Semantic similarity
Information extraction
Question answering
Summarization
Translation
Retrieval
Dialogue
```

## Information Extraction

```text
text
 ↓
NER
 ↓
entities
 ↓
relations / events
 ↓
structured JSON
```

Ví dụ:

```json
{
  "person": "Nam",
  "pickup": "A",
  "destination": "B",
  "time": "08:00"
}
```

---

# 20. Dialogue systems

```text
User utterance
 ↓
NLU / LLM understanding
 ↓
State tracking
 ↓
Dialogue policy / flow
 ↓
Action / tool
 ↓
Response generation
```

---

# 21. Rasa / dialogue management

Rasa hiện có hai hướng lớn:

```text
NLU-based assistants
+
CALM / FlowPolicy
```

FlowPolicy là state-machine định hướng business logic, quản lý state transitions và dialogue stack.

Official:

https://rasa.com/docs/reference/config/policies/flow-policy/

Policy overview:

https://rasa.com/docs/reference/config/policies/overview/

Slots:

https://rasa.com/docs/reference/primitives/slots/

Phải hiểu bản chất:

```text
NLU
↓
intent / entities
↓
tracker state
↓
policy / flow
↓
action
↓
slot
↓
response
```

Sau đó mapping sang:

```text
LLM Agent
↓
state
↓
tool
↓
graph / policy
```

---

# 22. LLM fine-tuning

Phân biệt:

```text
Prompting
RAG
SFT
PEFT
Preference tuning
Continued pretraining
```

Không phải mọi problem đều cần fine-tuning.

---

# 23. SFT

Supervised Fine-Tuning:

```text
Prompt
+
Desired response
       ↓
     Model
       ↓
 minimize loss
```

Dataset dạng chat:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

SFT thường dùng để điều chỉnh:

```text
instruction following
format
style
workflow behavior
domain task behavior
```

Hugging Face TRL:

https://huggingface.co/docs/trl/

SFTTrainer:

https://huggingface.co/docs/trl/sft_trainer

---

# 24. PEFT / LoRA

LoRA giữ base model frozen và học low-rank updates.

Khái niệm:

```text
W' = W + ΔW

ΔW ≈ A B
```

với:

```text
rank r << hidden dimensions
```

Mental model:

```text
Base weights
   │
 frozen
   │
 + LoRA adapters
   │
 train only adapters
```

PEFT docs:

https://huggingface.co/docs/peft/

LoRA guide:

https://huggingface.co/docs/peft/main/conceptual_guides/lora

Paper:

https://arxiv.org/abs/2106.09685

---

# 25. QLoRA

QLoRA:

```text
4-bit quantized base model
+
LoRA adapters
```

Mục tiêu:

```text
reduce GPU memory
```

Paper:

https://arxiv.org/abs/2305.14314

Cần hiểu:

- NF4
- double quantization
- paged optimizers
- frozen quantized model
- trainable adapter

---

# 26. Preference tuning / DPO

SFT:

```text
prompt → desired answer
```

DPO:

```text
prompt
 ├── chosen
 └── rejected
```

Paper:

https://arxiv.org/abs/2305.18290

Thứ tự nên học:

```text
SFT
 ↓
LoRA
 ↓
Evaluation
 ↓
DPO
```

---

# 27. Fine-tuning data engineering

Data quality thường quan trọng hơn việc tăng model size một cách mù quáng.

Dataset nên:

```text
consistent
correct
diverse
representative
deduplicated
safe
well-formatted
```

## Split

Không random split nếu có leakage từ:

```text
same document
same user
same template
near-duplicate
same conversation
```

## Hard examples

Nên có:

```text
ambiguous
long context
edge cases
negative examples
out-of-domain
code-switching
typos
speech-like text
```

## Speech/NLP domain data

Ví dụ:

```text
noise
accents
Vietnamese code-switching
street addresses
time expressions
self-corrections
hesitations
```

---

# 28. Fine-tuning evaluation

Không chỉ nhìn:

```text
training loss ↓
```

Phải có:

```text
task metrics
+
behavioral metrics
+
regression tests
+
safety tests
```

Ví dụ booking assistant:

```text
Intent Accuracy
Entity F1
JSON Validity
Slot Accuracy
Task Success Rate
Refusal correctness
Safety / policy compliance
```

---

# 29. Khi nào RAG tốt hơn fine-tuning?

### Knowledge thay đổi thường xuyên

```text
→ RAG
```

### Cần behavior / style / format mới

```text
→ SFT / PEFT
```

### Domain terminology

```text
→ RAG + prompting
→ fine-tuning nếu behavior vẫn chưa đủ
```

### Private knowledge

```text
→ RAG
```

### Consistent workflow

```text
→ SFT / PEFT
```

Thực tế có thể kết hợp:

```text
Base LLM
   +
RAG
   +
LoRA
   +
Tools
```

---

# 30. Track B — Computer Vision

CV là secondary specialization.

Mục tiêu:

> **Hiểu đủ sâu để xây vision-based AI systems và nối chúng với LLM.**

---

# 31. CV foundations

Cần biết:

- image tensor
- H × W × C
- RGB
- normalization
- augmentation
- convolution
- feature maps
- receptive field
- pretrained backbones
- transfer learning

Torchvision:

https://docs.pytorch.org/vision/stable/

---

# 32. Image segmentation

Segmentation:

```text
Image
 ↓
pixel-level predictions
```

### Semantic segmentation

Mỗi pixel nhận một class:

```text
road
car
person
sky
```

Không phân biệt individual instances cùng class.

### Instance segmentation

```text
person_1
person_2
car_1
car_2
```

mỗi object có mask riêng.

### Panoptic segmentation

Kết hợp:

```text
semantic
+
instance
```

---

# 33. Segmentation architectures

## U-Net

Paper:

https://arxiv.org/abs/1505.04597

Mental model:

```text
Encoder
   ↓
Bottleneck
   ↓
Decoder
```

với skip connections:

```text
encoder feature
       └──────→ decoder
```

## DeepLab / Torchvision

Torchvision có pretrained semantic segmentation models như:

```text
DeepLabV3
FCN
LRASPP
```

Docs:

https://docs.pytorch.org/vision/main/models

---

# 34. Segmentation evaluation

## IoU

```text
IoU = Intersection / Union
```

## Dice

```text
Dice = 2|A ∩ B| / (|A| + |B|)
```

Cần hiểu:

```text
pixel accuracy
IoU
Dice
precision
recall
```

Pixel accuracy có thể gây hiểu lầm khi class imbalance lớn.

---

# 35. SAM / SAM 2

SAM 2 là foundation model cho promptable visual segmentation trên images và videos.

Official repository:

https://github.com/Segment-Anything/segment-anything-2

Mental model:

```text
Image
 ↓
point / box / prompt
 ↓
mask
```

Video extension:

```text
frame_1
 ↓
mask
 ↓
memory
 ↓
frame_2
 ↓
...
```

Nên học SAM 2 để hiểu:

```text
promptable segmentation
foundation model
video segmentation
```

---

# 36. Track C — Vision-Language / Multimodal

Điểm nối:

```text
Computer Vision
+
LLM
```

---

# 37. VLM architecture

Một VLM thường có:

```text
Image
 ↓
Vision Encoder
 ↓
Visual Features
 ↓
Projector / Adapter
 ↓
LLM
 ↓
Text output
```

Ví dụ:

```text
Image
  ↓
Vision Transformer / Encoder
  ↓
Visual embeddings
  ↓
Projection
  ↓
Language model
  ↓
Text
```

## BLIP-2

Paper:

https://arxiv.org/abs/2301.12597

BLIP-2 bridge:

```text
frozen image encoder
+
Querying Transformer
+
frozen LLM
```

## LLaVA

Paper:

https://arxiv.org/abs/2304.08485

Mental model:

```text
Vision encoder
      ↓
Projection
      ↓
LLM
```

---

# 38. VLM inference

Hugging Face Transformers hiện hỗ trợ multimodal chat templates cho image, audio và video; `Processor` xử lý preprocessing/tokenization/chat template cho multimodal inputs.

Official:

https://huggingface.co/docs/transformers/main/chat_templating_multimodal

Conceptual input:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "..."},
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]
```

Pipeline:

```text
messages
 ↓
Processor
 ↓
input_ids + image tensors
 ↓
VLM
 ↓
generated tokens
```

---

# 39. VLM evaluation

Các task:

```text
image captioning
VQA
visual reasoning
OCR
object grounding
document understanding
chart understanding
```

Metrics phụ thuộc task:

```text
Exact match
BLEU / ROUGE
semantic similarity
VQA accuracy
OCR CER / WER
grounding IoU
task success
human evaluation
```

Không nên chỉ dùng benchmark score trong production.

---

# 40. CV + LLM integration

## Pattern 1 — CV first, LLM second

```text
Image
 ↓
OCR / Detection / Segmentation
 ↓
Structured result
 ↓
LLM
 ↓
Answer
```

## Pattern 2 — VLM direct

```text
Image + Question
 ↓
VLM
 ↓
Answer
```

## Pattern 3 — Hybrid

```text
             Image
               │
       ┌───────┴───────┐
       ▼               ▼
     CV/OCR            VLM
       │               │
       └───────┬───────┘
               ▼
              LLM
               │
               ▼
            Answer
```

Rule of thumb:

```text
Need explicit, debuggable structure?
→ CV/OCR + LLM

Need direct visual reasoning?
→ VLM

Need both?
→ Hybrid
```

---

# 41. Document AI / OCR + LLM

Một pipeline thực tế:

```text
PDF / Image
    ↓
Layout detection
    ↓
OCR
    ↓
Tables / blocks / figures
    ↓
Structured representation
    ↓
Embedding / retrieval
    ↓
LLM
```

Phải chú ý:

```text
reading order
tables
headers / footers
multi-column layout
charts
scanned PDFs
handwriting
```

OCR sai thì downstream reasoning có thể sai.

---

# 42. Audio + LLM / speech-language models

## Cascaded

```text
Audio
 ↓
ASR
 ↓
Text LLM
 ↓
TTS
```

Ưu:

- dễ debug
- dễ evaluate từng stage
- dễ thay component

## Native multimodal

```text
Audio
 ↓
Speech / multimodal model
 ↓
Text / action / audio
```

Có thể giữ information chưa được biểu diễn hoàn toàn bằng text, nhưng khó hơn về training và evaluation.

---

# 43. Optional Track — Recommendation Systems

Chỉ học nếu JD/product cần.

Pipeline:

```text
User
+
Item
+
Interaction
      ↓
Candidate generation
      ↓
Filtering
      ↓
Ranking
      ↓
Top-N
```

Cần biết:

- collaborative filtering
- content-based recommendation
- matrix factorization
- embeddings
- candidate generation
- ranking
- cold start
- offline evaluation
- A/B testing

---

# 44. Hands-on Project 1 — Production Speech Pipeline

## Goal

```text
Microphone / Audio file
        ↓
VAD
        ↓
ASR
        ↓
timestamps
        ↓
speaker diarization
        ↓
text normalization
        ↓
NLP / NER / intent
        ↓
structured JSON
```

## Stack

```text
Python
PyTorch
Whisper / faster-whisper
CTranslate2
pyannote.audio
Hugging Face Transformers
FastAPI
Docker
```

## Output

```json
{
  "language": "vi",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 1.2,
      "end": 4.8,
      "text": "Tôi muốn đặt xe"
    }
  ],
  "intent": "book_taxi",
  "entities": {
    "pickup": null,
    "destination": null
  }
}
```

## Benchmark

```text
WER
CER
entity accuracy
intent accuracy
latency
RTF
CPU memory
GPU memory
```

---

# 45. Hands-on Project 2 — Fine-tuned Domain LLM

## Goal

Fine-tune model nhỏ cho:

```text
domain-specific assistant
```

## Experiment ladder

### Step 1 — Baseline

```text
base model + prompt
```

### Step 2 — RAG

```text
base model + RAG
```

### Step 3 — LoRA / SFT

```text
base model + LoRA
```

### Step 4 — Optional DPO

```text
SFT model
 ↓
preference data
 ↓
DPO
```

### Step 5 — Compare

```text
Base
vs
RAG
vs
LoRA
vs
RAG + LoRA
vs
RAG + LoRA + DPO
```

Đây là project rất tốt để hiểu khi nào adaptation thực sự có value.

---

# 46. Hands-on Project 3 — Dialogue / Voice Assistant

## Architecture

```text
Microphone
    ↓
VAD
    ↓
STT
    ↓
Dialogue / Agent
    │
    ├── RAG
    ├── Booking API
    ├── Database
    └── MCP tools
    ↓
Response text
    ↓
TTS
    ↓
Speaker
```

## Dialogue state

```json
{
  "intent": "book_taxi",
  "pickup": "Vinhome Riverside",
  "destination": null,
  "time": "08:00",
  "vehicle_type": "normal_car"
}
```

## Error recovery

Phải test:

```text
ASR wrong
user correction
missing slot
ambiguous slot
user changes mind
tool timeout
LLM hallucination
TTS failure
```

---

# 47. Hands-on Project 4 — Segmentation + VLM

## Use case

Upload ảnh và hỏi AI về vùng/object trong ảnh.

Pipeline:

```text
Image
 ↓
SAM 2 segmentation
 ↓
Objects / masks
 ↓
Crop / visual features
 ↓
VLM
 ↓
LLM reasoning
 ↓
structured response
```

Evaluation:

```text
mask IoU
segmentation quality
VLM answer accuracy
latency
failure rate
```

---

# 48. Capstone — Multimodal Voice AI Assistant

Đây là project portfolio chính sau Tầng 3.

## Use case

> Voice AI assistant cho taxi booking.

User:

```text
"Mai 8 giờ sáng đón tôi ở Vinhome Riverside đến 47 Trần Khát Chân."
```

Pipeline:

```text
                     USER SPEECH
                          │
                          ▼
                    ┌──────────┐
                    │   VAD    │
                    └────┬─────┘
                         ▼
                    ┌──────────┐
                    │   STT    │
                    │ Whisper  │
                    └────┬─────┘
                         ▼
                  ┌───────────────┐
                  │ NLP / Agent   │
                  │ LangGraph     │
                  └──────┬────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
       RAG              MCP             Database
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                        LLM
                         │
                         ▼
                  Structured result
                         │
                         ▼
                        TTS
                         │
                         ▼
                     USER AUDIO
```

## Optional multimodal extension

User có thể nói và đưa ảnh:

```text
"Đón tôi ở chỗ này"
+
photo
```

Pipeline:

```text
Voice
 ↓
STT
 ↓
Image
 ↓
VLM / CV
 ↓
visual context
 ↓
Agent
 ↓
Booking tool
```

## Production requirements

```text
FastAPI
Docker
RAG
Vector DB
Agent
MCP
STT
TTS
Logging
Tracing
Evaluation
Auth
Rate limiting
timeouts
```

---

# 49. Interview checklist

## Speech

> WER khác CER thế nào?

> CTC hoạt động thế nào?

> RNN-T khác CTC thế nào?

> Whisper khác CTC ASR thế nào?

> Conformer thêm convolution để làm gì?

> VAD có tác động gì đến ASR cost?

> Diarization giải quyết gì?

> Forced alignment khác ASR timestamps thế nào?

> RTF là gì?

> Streaming ASR khác offline ASR thế nào?

> Tại sao WER tốt nhưng business entity vẫn sai?

## TTS

> Acoustic model và vocoder khác nhau thế nào?

> Mel spectrogram là gì?

> MOS là gì?

> Làm sao đo speaker similarity?

> Làm sao kiểm tra intelligibility?

## NLP / Dialogue

> NLU khác NLG thế nào?

> Dialogue state là gì?

> Slot filling là gì?

> Dialogue policy làm gì?

> Flow-based dialogue khác free-form LLM agent thế nào?

> Làm sao xử lý user correction?

## Fine-tuning

> Khi nào dùng RAG?

> Khi nào dùng SFT?

> LoRA hoạt động thế nào?

> QLoRA khác LoRA thế nào?

> Fine-tuning dataset tốt là gì?

> Làm sao tránh data leakage?

> DPO khác SFT thế nào?

> Tại sao loss giảm nhưng task performance không tăng?

## Computer Vision

> Semantic và instance segmentation khác nhau thế nào?

> IoU và Dice khác nhau thế nào?

> U-Net dùng skip connections để làm gì?

> SAM 2 khác U-Net thế nào?

## VLM

> Vision encoder làm gì?

> Projector làm gì?

> Làm thế nào đưa visual representations vào LLM?

> VLM khác OCR + LLM thế nào?

> Khi nào chọn CV + LLM thay vì VLM trực tiếp?

---

# 50. Bộ tài liệu ưu tiên

## Speech

### 1. Whisper
https://github.com/openai/whisper

### 2. Hugging Face Audio Course
https://huggingface.co/learn/audio-course/chapter0/introduction

### 3. Wav2Vec 2.0
https://arxiv.org/abs/2006.11477

### 4. Conformer
https://arxiv.org/abs/2005.08100

### 5. NVIDIA NeMo Speech
https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html

### 6. NeMo Tutorials
https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html

### 7. pyannote.audio
https://github.com/pyannote/pyannote-audio

### 8. CTranslate2
https://github.com/OpenNMT/CTranslate2

### 9. faster-whisper
https://github.com/SYSTRAN/faster-whisper

### 10. TorchAudio forced alignment
https://docs.pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html

---

## NLP / LLM

### 11. Hugging Face Transformers
https://huggingface.co/docs/transformers/

### 12. Hugging Face TRL
https://huggingface.co/docs/trl/

### 13. Hugging Face PEFT
https://huggingface.co/docs/peft/

### 14. LoRA
https://arxiv.org/abs/2106.09685

### 15. QLoRA
https://arxiv.org/abs/2305.14314

### 16. DPO
https://arxiv.org/abs/2305.18290

---

## Dialogue

### 17. Rasa FlowPolicy
https://rasa.com/docs/reference/config/policies/flow-policy/

### 18. Rasa Policy Overview
https://rasa.com/docs/reference/config/policies/overview/

### 19. Rasa Slots
https://rasa.com/docs/reference/primitives/slots/

---

## Computer Vision

### 20. Torchvision
https://docs.pytorch.org/vision/stable/

### 21. U-Net
https://arxiv.org/abs/1505.04597

### 22. SAM 2
https://github.com/Segment-Anything/segment-anything-2

### 23. MMDetection
https://github.com/open-mmlab/mmdetection

---

## Multimodal

### 24. Hugging Face Multimodal Chat Templates
https://huggingface.co/docs/transformers/main/chat_templating_multimodal

### 25. BLIP-2
https://arxiv.org/abs/2301.12597

### 26. LLaVA
https://arxiv.org/abs/2304.08485

---

# 51. Skill matrix

| Skill | Mức cần đạt |
|---|---|
| Audio fundamentals | Strong |
| ASR fundamentals | Strong |
| Whisper architecture | Strong |
| Whisper inference | Strong |
| CTranslate2 / faster-whisper | Practical |
| CTC | Strong |
| RNN-T / Transducer | Working knowledge |
| Conformer | Strong |
| VAD | Practical |
| Forced alignment | Practical |
| Speaker diarization | Practical |
| ASR evaluation | Strong |
| TTS fundamentals | Strong |
| TTS inference | Practical |
| TTS evaluation | Working knowledge |
| NLP fundamentals | Strong |
| Dialogue management | Strong |
| Rasa | Strong |
| LLM SFT | Strong |
| PEFT / LoRA | Strong |
| QLoRA | Practical |
| DPO | Working knowledge |
| Fine-tuning data | Strong |
| Fine-tuning evaluation | Strong |
| Semantic segmentation | Strong |
| Instance segmentation | Working knowledge |
| U-Net | Strong |
| SAM 2 | Practical |
| VLM architecture | Strong |
| VLM inference | Practical |
| CV + LLM integration | Strong |
| Multimodal evaluation | Practical |

---

# 52. Definition of Done

## Speech

Bạn phải có thể:

```text
audio
 ↓
VAD
 ↓
ASR
 ↓
timestamps
 ↓
diarization
 ↓
NLP
```

và debug được:

```text
wrong transcript
wrong language
hallucination
wrong timestamps
wrong speaker
slow inference
```

## Fine-tuning

Bạn phải có thể:

```text
baseline
 ↓
RAG baseline
 ↓
SFT
 ↓
LoRA
 ↓
evaluation
 ↓
ablation
```

và giải thích:

```text
Why fine-tuning?
Why not RAG?
Why this dataset?
Why LoRA?
Why these target modules?
Why this metric?
```

## Dialogue

Bạn phải có thể thiết kế:

```text
NLU
 ↓
state
 ↓
policy
 ↓
tool
 ↓
response
```

và xử lý:

```text
correction
clarification
confirmation
fallback
timeout
interrupt
```

## Vision

Bạn phải có thể:

```text
image
 ↓
segmentation
 ↓
mask evaluation
```

và hiểu:

```text
semantic
vs
instance
vs
panoptic
```

## VLM

Bạn phải có thể:

```text
image
+
text
 ↓
processor
 ↓
vision encoder
 ↓
projection
 ↓
LLM
 ↓
answer
```

và biết khi nào chọn:

```text
CV + LLM
vs
VLM
```

---

# 53. Roadmap học theo thứ tự

## Stage 1 — Speech foundations

```text
Waveform
 ↓
Sampling
 ↓
STFT
 ↓
Mel spectrogram
 ↓
VAD
```

Project:

```text
Audio preprocessing notebook
```

## Stage 2 — ASR

```text
CTC
 ↓
Wav2Vec2
 ↓
Whisper
 ↓
Decoding
 ↓
WER / CER
```

Project:

```text
ASR benchmark
```

## Stage 3 — Production speech

```text
faster-whisper
 ↓
CTranslate2
 ↓
quantization
 ↓
batching
 ↓
timestamps
 ↓
diarization
```

Project:

```text
Production STT service
```

## Stage 4 — TTS

```text
text normalization
 ↓
acoustic model
 ↓
vocoder
 ↓
waveform
```

Project:

```text
Voice synthesis service
```

## Stage 5 — Dialogue

```text
NLU
 ↓
state
 ↓
policy
 ↓
tool
 ↓
response
```

Project:

```text
Voice booking assistant
```

## Stage 6 — LLM Fine-tuning

```text
Baseline
 ↓
dataset
 ↓
SFT
 ↓
LoRA
 ↓
evaluation
 ↓
optional DPO
```

Project:

```text
Domain-specific assistant
```

## Stage 7 — Vision

```text
image tensor
 ↓
CNN / ViT
 ↓
segmentation
 ↓
IoU / Dice
 ↓
SAM 2
```

Project:

```text
Image segmentation application
```

## Stage 8 — Multimodal

```text
image
+
text
 ↓
VLM
 ↓
reasoning
```

Project:

```text
Vision + LLM assistant
```

## Stage 9 — Capstone

```text
Speech
+
NLP
+
LLM
+
RAG
+
Agent
+
MCP
+
TTS
+
optional Vision
+
FastAPI
+
Docker
```

---

# Final specialization strategy

Không nên cố trở thành:

```text
"biết sơ sơ 20 domain"
```

Nên hướng đến:

```text
Strong Core
    +
Strong Speech/NLP
    +
Production AI Engineering
    +
Working Multimodal Knowledge
```

Profile cuối cùng:

```text
             AI / ML CORE
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   PRODUCTION          DOMAIN EXPERTISE
        │                   │
   RAG / Agent          Speech / NLP
   MCP                  LLM Fine-tuning
   FastAPI              Dialogue
   Docker               ASR / TTS
   Serving                   │
        │                 Multimodal
        └─────────┬─────────┘
                  ▼
          SPECIALIZED AI
             ENGINEER
```

Mục tiêu cuối cùng không phải là:

> **"Biết dùng Whisper, Rasa, Hugging Face hay VLM."**

Mà là:

> **Nhìn một bài toán domain và biết chọn architecture, data strategy, model, adaptation method, evaluation metrics và deployment strategy phù hợp.**
