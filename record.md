# 记录一些自己的思考

## Llama2推理过程中的数据shape变换
### 一些预定义参数
- max_q_len: 仅限当前轮次中，所有batch中问题的最大长度
- max_k_len: 到当前轮次为止，不同batch中上下文长度的最大值
- max_seq_len: 在一次考虑上下文的推理过程中, 最大能支持的上下文长度
- hidden_size: 每个token的embedding长度
- q_head_num: query的 注意力头数
- kv_head_num: key 和 value的注意力头数
- num_tokens: 所有batch的tokens总数目

### Context Decoder中的形状变化
1. Embedding
- 将输入的token id从embedding_table中进行查找, 找到对应token的embedding
- [num_token] --> [num_token, hidden_size]

2. CalPaddingOffset
- 构造PaddingOffset, 每个offset代表对应位置的token其前面padding的token的数目。举例如下:
11100       00022
11000 ===>  55555
11111       [00000]    最后一行的0代表未填充，其值取决于分配空间时的初始化的值

- shape of paddingOffset:  *[batch_size, max_q_len]*

3. Build Causal Mask
- 下三角矩阵主要用来mask一些计算结果，保证每个token的计算结果只考虑其前面的token，不能让token获取其后置token的信息。具体为什么这么做，我理解中是要和训练时保持一致。

- shape of Causal Mask: [batch_size, max_q_len, max_k_len], 即对于每一个batch的输入句子，我们要计算句子中的每个token(max_q_len)对其上下文中每个token的掩码。对于每个[max_q_len, max_k_len]的矩阵，其元素Ture所构成的形状为梯形, 上底长度为 `max_k_len - max_q_len`, 下底长度为 `max_k_len`

4. RMSNorm
- 对输入的每个token在其embedding维度做一个归一化, RMSNorm的计算公式如下：
`x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`

- shape变换: [num_tokens, hidden_size] ===> [num_tokens, hidden_size]

5. Qkv Gemm
- 对输入的经过归一化的embedding做维度变换，生成三个矩阵，分别代表维度变换后每个token在Query Key 和 Value层面的值

- 这里我们做一个矩阵融合，即将WQ WK WV 三个权重矩阵横向拼接, 将三个矩阵乘法融合为一个大的矩阵乘法, 以实现高效计算

- shape变化: [num_tokens, hidden_size] ===> [num_tokens, qkv_hidden_size]，其中qvk_hidden_size = q_hidden_size + k_hidden_size + v_hidden_size = q_hidden_size + 2 * k_hidden_size

6. AddBias_Padding_Rope
- 这个算子是融合算子, 为每个embedding添加bias, 然后通过padding_offset将num_tokens 还原到 [bs, max_q_len]这样一个矩阵形状。其本质上只是为每个token找到其矩阵存储形式下的对应位置，然后写入对应位置即可。其他已分配但未写入的地方即初始化的值

- 在对应位置写入embedding之前, 我们需要对embedding添加位置编码，即进行RoPe(Rotary Position Embedding)编码, 其中在计算位置编码时我们需要参数m, 即当前embedding的位置，这个位置是相对历史上下文的位置(当前batch的历史上下文token_size + 该token在当前句子中的位置)

- shape变化: [num_tokens, qvk_hidden_size] ==> [bs, head_num, max_q_len, k_head_size] + [bs, head_num, max_q_len, q_head_size] + [bs, head_num, max_q_len, v_head_size]

7. Concat KV
- 这里主要是实现KV Cache, 即计算过程中添加位置信息的K以及经过维度变换的V都是可以cache下来实现复用的. k和v cache的shape为: [layer_num, batch_size, head_num, max_seq_len, head_size]. (这样看来如何组织不同的batch也是一个问题, 所以如果将推理服务进行部署，那么其kv cache的存储肯定会更加复杂.....)

- 至于拼接过程就是一个典型的数据读写过程, 从kv中(shape:[bs, kv_head_num, max_q_len, head_size]) 读取对应数据, 写到 kv_cache(如上所示)中的对应位置。需要注意的是kv_cache中的max_seq_len维度依然需要考虑上下文。

8. Repeat KV
- Repeat KV主要是为了实现KV的复用，将kv(kv_head_num)的维度广播为q(q_head_num) 的维度, 用于MQA(Multi-Query Attention) 和 GQA(Group-Query Attention), 这两个措施都是为了减少kv cache的大小从而实现对更大batch的数据推理(当然肯定也有利于训练), 其核心思想即为多个query共享一组kv, 如果所有query共享一组kv即为MQA, 分组共享即为GQA, 不共享即为Transformer经典的MHA。


- 我们需要将当前已经持有的所有kv cache都进行 repeat, 即输出的kv的shape为：[bs, q_head_num, max_k_len, head_size]

9. Qk Gemm
- 这里就是进行矩阵运行计算每个token对其过往token未经归一化的attention score

- 其输出shape为: [bs, q_head_num, max_q_len, max_k_len]

10. Scale_AttentionMask_SoftMax
- AttentionMask shape: [bs, max_q_len, max_k_len], 即上文提到的下三角矩阵
- softmax计算公式：(待补充)
- softmax的问题: 因为涉及指数次方, 很大概率会溢出, 所以我们在求指数之前要先将每个数减去该行中的最大值, 然后在求指数


11. Qk * v Gemm

- 输出形状 [bs, q_head_hum, max_q_len, head_size]

12. Remove Padding 
- Remove Padding 即将输入token的矩阵形式还原为连续存储的形式, 即去除padding
- [bs, q_head_num, max_q_len, head_size] => [num_token, q_hidden_size]
- 具体去除padding也是一个读写的过程，应该是要用到不同句子长度的前缀和数组, 来判断每个batch中前多少个token为非padding token

13. Output Linear
- 形状还原
- shape变化: [num_token, q_hidden_size] ==> [num_token, hidden_size]

14. 一些细节:
- 在context decoder中, 每个轮次我们只计算当前轮次输入的token的attention,只是要考虑历史上下文而已。

### Self Decoder中的形状变化
