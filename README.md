# Attention-seq2seq_Transformers

Attention Mechanisms
Role of Q, K, V: Query (Q) represents the input's current state. Key (K) encodes all inputs' states. Value (V) contains information related to keys. Attention scores are computed as 
softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
softmax( 
d 
k
​
 
​
 
QK 
T
 
​
 ), which are used to weight 
𝑉
V for focusing on relevant parts.
Scale Factor: 
𝑑
𝑘
d 
k
​
 
​
  prevents the dot product of Q and K from growing too large, avoiding vanishing gradients when passed through softmax.
Multi-Head Attention: Enables the model to focus on multiple aspects of the input (e.g., syntax and semantics) by splitting Q, K, V into smaller dimensions processed independently and concatenated.
Masking: Prevents the model from attending to future tokens in sequence generation, ensuring causality.
Self-Attention vs Cross-Attention: Self-attention attends to the same input sequence; cross-attention attends to another sequence (e.g., encoder-decoder tasks).
Relative Positional Encoding: Explicitly represents the distance between tokens, unlike absolute positional encoding, making it better for tasks requiring long-term dependencies.
Positional Encoding
Why Positional Encoding? Transformers lack recurrence, so positional encoding adds sequence order information to the model.
Sinusoidal Encoding: Uses sine and cosine functions to provide unique values based on position and dimension:
𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
)
=
sin
⁡
(
𝑝
𝑜
𝑠
1000
0
2
𝑖
/
𝑑
)
,
𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
+
1
)
=
cos
⁡
(
𝑝
𝑜
𝑠
1000
0
2
𝑖
/
𝑑
)
PE 
(pos,2i)
​
 =sin( 
10000 
2i/d
 
pos
​
 ),PE 
(pos,2i+1)
​
 =cos( 
10000 
2i/d
 
pos
​
 )
Alternatives: Learnable embeddings or rotary positional embeddings (RoPE) allow for greater adaptability to the task.
Encoder-Decoder Architecture
Why Encoder-Decoder? For sequence-to-sequence tasks, the encoder captures source sequence information, and the decoder generates the target sequence.
Masked Multi-Head Attention: Ensures the decoder only attends to previous tokens, maintaining autoregressive properties.
Encoder-Decoder Attention: Allows the decoder to focus on the most relevant parts of the encoded source sequence.
Transformer Variants
BERT vs GPT:
BERT: Bidirectional, uses Masked Language Modeling (MLM).
GPT: Autoregressive, generates text token-by-token.
Autoencoding vs Autoregressive: Autoencoding models (e.g., BERT) encode the full context; autoregressive models predict sequentially.
T5: Unified framework for all NLP tasks, converts tasks into a text-to-text format.
ViT: Splits images into patches and processes them like tokens in text.
Transformer-XL: Adds recurrence to capture long-term dependencies.
Longformer: Employs sparse attention for efficiency with long sequences.
Swin Transformer: Hierarchical structure with shifted windows for image tasks.
Optimization and Training
Label Smoothing: Prevents the model from becoming overconfident, regularizing predictions.
Gradient Clipping: Limits gradient magnitude to prevent exploding gradients.
Adam with Warm-up: Addresses instability during initial training by gradually increasing the learning rate.
Long Sequence Handling: Efficient architectures like Reformer, Sparse Transformer, and Longformer reduce computational costs.
Applications of Transformers
Translation Tasks: Encoder-decoder with cross-attention aligns source and target sequences.
Summarization Models: Use pretrained Transformers like Pegasus with task-specific fine-tuning.
Speech Recognition: Use models like Speech-Transformer; audio frames are treated as tokens.
Real-Time Applications: Techniques like pruning, quantization, and distillation reduce latency.
Advanced Mechanisms
Sparsity in Attention: Reduces computational complexity by focusing on specific token pairs.
Mixture of Experts: Activates only parts of the model, improving scalability.
Pretraining and Fine-Tuning: Pretrained on massive corpora, fine-tuned on specific tasks to adapt to new domains.
Code Examples
Multi-Head Self-Attention: Python implementation of scaled dot-product attention.
Encoder Block: Write code for multi-head attention, positional encoding, and feed-forward layers.
Causal Masking: Add masks to attention scores before applying softmax.
