# This file contains the model structure information we need to compute flops

max_seqL = 1 for decode stages
tp = tensor parallel degree


LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): VocabParallelEmbedding()          X = {seqN, max_seqL}; W = {pad(V, 64)/tp, h}:  O(seqN * max_seqL * h)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(                 
        (self_attn): LlamaAttention(
          (qkv_proj): QKVParallelLinear()             X = {seqN, max_seqL, h}; W = {h, (head_num/tp_size+2*max(kv_head_num/tp_size,1))*head_dim}: O(seqN * max_seqL * h * ((head_num/tp_size+2*max(kv_head_num/tp_size,1))*head_dim))
          (o_proj): RowParallelLinear()               X = {seqN, max_seqL, h}; W = {h/tp_size, h}: O(seqN * max_seqL * h * h)
          (rotary_emb): RotaryEmbedding()             X = {seqN, max_seqL, h}; cos_sin_cache = {max_pos, head_dim}: O(seqN * max_seqL * h) or O(2 * seqN * max_seqL * h) [more precise] flops应该是和tp_size有关的
          (attn): PagedAttention()                    X = {seqN, max_seqL, h}: O(2*seqN*max_seqL*h*max_seqL) 这里直接max_seqL平方好像不太对 for old flash attention prefill; in the latest flash attention prefill, var len is supported; 
        )
        (mlp): LlamaMLP(
          (gate_up_proj): MergedColumnParallelLinear()  X = {seqN, max_seqL, h}; W = {h, 2*I/tp_size}: O(seqN*max_seqL*h*2*I)
          (down_proj): RowParallelLinear()              X = {seqN, max_seqL, I}; W = {I/tp_size, h}: O(seqN*max_seqL*I*h)
          (act_fn): SiluAndMul()                        X = {seqN, max_seqL, 2*I}: O(seqN*max_seqL*I) [omit complexity for silu() here] flops应该和tp size有关
        )
        (input_layernorm): RMSNorm()                    X = {seqN, max_seqL, h}; W = {h}: O(seqN*max_seqL*h) [omit constant coefficient (2?)]
        (post_attention_layernorm): RMSNorm()           X = {seqN, max_seqL, h}; W = {h}: O(seqN*max_seqL*h) [omit constant coefficient (2?)]
      )
    )
    (norm): RMSNorm()                                   X = {seqN, max_seqL, h}; W = {h}: O(seqN*max_seqL*h) [omit constant coefficient (2?)]
  )
  (lm_head): ParallelLMHead()                           W = {pad(V, 64)/tp, h}: [this layer is not used] the weights are used to get logits via matmul
  (sampler): Sampler()
)


TOTAL COMPLEXITY:
B = seqN

for PREFILL:
(1) xops.memory_efficient_attention_forward里的attention_bias参数通过对QK^T+bias的方式对将bias设成0或者-inf,达到mask的目的。
flops = B*max_seqL*h + L*( 4*B*max_seqL*h*h + 2*B*max_seqL*h + ``2*B*max_seqL*max_seqL*h'' + 3*B*max_seqL*h*I + B*max_seqL*I + 2*B*max_seqL*h) + B*max_seqL*h
==>
s = max_seqL
flops = B*s*h + L*( 4*B*s*h*h + 2*B*s*h + ``2*B*s*s*h'' + 3*B*s*h*I + B*s*I + 2*B*s*h) + B*s*h
~ L*( 4*B*s*h*h + ``2*B*s*s*h'' + 3*B*s*h*I)

考虑了tp_size 之后的flops:
flops = B*s*h + L*( B*s*h*[(head_num/tp_size+2*max(kv_head_num/tp_size,1))*head_dim] + 
                   B*s*h/tp_size*h + 
                   B*s*(h+kv_head_num*head_dim)/tp_size + 
                   ``2*B*s*h*s/tp_size'' + 
                   B*s*h*2*I/tp_size + 
                   B*s*I/tp_size*h + 
                   B*s*I/tp_size + 
                   B*s*h*2) + B*s*h
~ L*( B*s*h*[(2*max(kv_head_num/tp_size,1))*head_dim] + 
                   2*B*s*h/tp_size*h +  
                   ``2*B*s*h*s/tp_size'' + 
                   3*B*s*I/tp_size*h)



(2) flash_attn.flash_attn_varlen_func这一函数
for the latest vllm which supports var_len prefill flash attention:
flops = sum(si)*h + L*( 4*sum(si)*h*h + 2*sum(si)*h + ``sum_{i=1,...,B}(h*(1+si)*si)'' + 3*sum(si)*h*I + sum(si)*I + 2*sum(si)*h) + sum(si)*h
==>
~ L*( 4*sum(si)*h*h + ``sum_{i=1,...,B}(h*(1+si)*si)'' + 3*sum(si)*h*I)    [approximate]
= L*( 4*sum(si)*h*h + ``h*sum((1+si)*si)'' + 3*sum(si)*h*I)

考虑了tp_size 之后的flops:
~ L*( sum(si)*h*[(2*max(kv_head_num/tp_size,1))*head_dim] + 
                   2*sum(si)*h/tp_size*h +  
                   ``h*sum((1+si)*si)/tp_size'' + 
                   3*sum(si)*I/tp_size*h)



for DECODE:
max_seqL = 1
flops = B*max_seqL*h + L*( 4*B*max_seqL*h*h + 2*B*max_seqL*h + ``sum_{i=1,...,B}(2*si*h)'' + 3*B*max_seqL*h*I + B*max_seqL*I + 2*B*max_seqL*h) + B*max_seqL*h
==>
s = max_seqL = 1      [it is for the query token per sequence; si is for the totol token for every sequence]
flops = 2*B*s*h + L*( 4*B*s*h*h + ``sum_{i=1,...,B}(2*si*h)'' + (3*I+4)*B*s*h + B*s*I)
~ L*( 4*B*s*h*h + ``sum_{i=1,...,B}(2*si*h)'' + 3*I*B*s*h)    [approximate]
= L*( 4*B*s*h*h + ``2*h*sum(si)'' + 3*I*B*s*h)


考虑了tp_size 之后的flops:
~ L*( B*s*h*[(2*max(kv_head_num/tp_size,1))*head_dim] + 
                   2*B*s*h/tp_size*h +  
                   ``2*h*sum(si)/tp_size'' + 
                   3*B*s*I/tp_size*h)


*******
pad成相同seq_len的情况下的prefill阶段的attention开销: 2*B*max_seqL*max_seqL*h
var_len的情况下的prefill阶段的attention的开销: sum_{i=1,...,B}(h*(1+si)*si)   [si is the sequence length for each sequence]
decoding阶段的attention的开销: sum_{i=1,...,B}(2*si*h)  [si is the sequence length for each sequence]




# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


ChatGLMForCausalLM(
  (transformer): ChatGLMModel(
    (embedding): VocabParallelEmbedding()
    (encoder): GLMTransformer(
      (layers): ModuleList(
        (0-27): 28 x GLMBlock(
          (input_layernorm): RMSNorm()
          (self_attention): GLMAttention(
            (query_key_value): QKVParallelLinear()
            (dense): RowParallelLinear()
            (rotary_emb): RotaryEmbedding()                       X = {seqN, max_seqL, h/tp_size}; cos_sin_cache = {max_pos, head_dim//2}: O(seqN * max_seqL * num_head * rot_dim) or O(2 * seqN * max_seqL * h) [more precise] flops应该是和tp_size有关的
            (attn): PagedAttention()
          )
          (post_attention_layernorm): RMSNorm()
          (mlp): GLMMLP(
            (dense_h_to_4h): MergedColumnParallelLinear()         config.ffn_hidden_size = intermediate_size
            (activation_func): SiluAndMul()
            (dense_4h_to_h): RowParallelLinear()
          )
        )
      )
      (final_layernorm): RMSNorm()
    )
    (output_layer): ParallelLMHead()
  )
  (sampler): Sampler()
)

NOTE: 和Llama系列模型不同的地方:
flops: (1) rotary embedding 的 rot dim 不同; (2) intermediate size的名称不同
parameter size: (1) rotary embedding 的 rot dim 不同; (2) 含有 qkv bias; (3) intermediate size的名称不同


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

GPTJForCausalLM(
  (transformer): GPTJModel(
    (wte): VocabParallelEmbedding()
    (h): ModuleList(
      (0-27): 28 x GPTJBlock(
        (ln_1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attn): GPTJAttention(
          (qkv_proj): QKVParallelLinear()
          (out_proj): RowParallelLinear()
          (rotary_emb): RotaryEmbedding()
          (attn): PagedAttention()
        )
        (mlp): GPTJMLP(
          (fc_in): ColumnParallelLinear()
          (fc_out): RowParallelLinear()
          (act): NewGELU()
        )
      )
    )
    (ln_f): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): ParallelLMHead()
  (sampler): Sampler()
)


NOTE: 和Llama系列模型不同的地方:
inner_dim->intermediate size
n_embd->hidden_size
flops: (1) rotary dim 来自config.rotary_dim (2) fc_in, fc_out 参数不同
parameter size: (1) ln_1,ln_f 有weight和bias参数 (2) rotary dim 来自config.rotary_dim
                (3) fc_in, fc_out 参数不同, 且有bias, bias 受tp_size影响
                (4) lm_head 有bias, 且bias也受tp_size影响




# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


GPTNeoXForCausalLM(
  (gpt_neox): GPTNeoXModel(
    (embed_in): VocabParallelEmbedding()
    (layers): ModuleList(
      (0-43): 44 x GPTNeoXLayer(
        (input_layernorm): LayerNorm((6144,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((6144,), eps=1e-05, elementwise_affine=True)
        (attention): GPTNeoXAttention(
          (query_key_value): QKVParallelLinear()
          (dense): RowParallelLinear()
          (rotary_emb): RotaryEmbedding()
          (attn): PagedAttention()
        )
        (mlp): GPTNeoXMLP(
          (dense_h_to_4h): ColumnParallelLinear()
          (dense_4h_to_h): RowParallelLinear()
          (act): FastGELU()
        )
      )
    )
    (final_layer_norm): LayerNorm((6144,), eps=1e-05, elementwise_affine=True)
  )
  (embed_out): ParallelLMHead()
  (sampler): Sampler()
)





# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
BaichuanForCausalLM(
  (model): BaiChuanModel(
    (embed_tokens): VocabParallelEmbedding()
    (layers): ModuleList(
      (0-39): 40 x BaiChuanDecoderLayer(
        (self_attn): BaiChuanAttention(
          (W_pack): QKVParallelLinear()
          (o_proj): RowParallelLinear()
          (attn): PagedAttention()
        )
        (mlp): BaiChuanMLP(
          (gate_up_proj): MergedColumnParallelLinear()
          (down_proj): RowParallelLinear()
          (act_fn): SiluAndMul()
        )
        (input_layernorm): RMSNorm()
        (post_attention_layernorm): RMSNorm()
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): ParallelLMHead()
  (sampler): Sampler()
)




# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
MixtralForCausalLM(
  (model): MixtralModel(
    (embed_tokens): VocabParallelEmbedding()
    (layers): ModuleList(
      (0-31): 32 x MixtralDecoderLayer(
        (self_attn): MixtralAttention(
          (qkv_proj): QKVParallelLinear()
          (o_proj): RowParallelLinear()
          (rotary_emb): RotaryEmbedding()
          (attn): PagedAttention()
        )
        (block_sparse_moe): MixtralMoE(
          (gate): ReplicatedLinear()
          NOTE: there are two more fused_moe_kernel
        )
        (input_layernorm): RMSNorm()
        (post_attention_layernorm): RMSNorm()
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): ParallelLMHead()
  (sampler): Sampler()
)