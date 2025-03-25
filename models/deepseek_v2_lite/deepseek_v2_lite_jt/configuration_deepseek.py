
class DeepseekV2Config():

    model_type = "deepseek_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size = 1408,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=16,
        n_shared_experts = 2,
        n_routed_experts = 64,
        ep_size = 1,
        routed_scaling_factor = 1.0,
        kv_lora_rank = 512,
        q_lora_rank = None,
        qk_rope_head_dim = 64,
        v_head_dim = 128,
        qk_nope_head_dim = 128,
        topk_method = 'greedy',
        n_group = 1,
        topk_group = 1,
        num_experts_per_tok = 6,
        moe_layer_freq = 1,
        first_k_dense_replace = 1,
        norm_topk_prob = False,
        scoring_func = 'softmax',
        aux_loss_alpha = 0.001,
        seq_aux = True,
        hidden_act="silu",
        max_position_embeddings=163840,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100000,
        eos_token_id=100001,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        if rope_scaling is None:
            rope_scaling = {
                'beta_fast': 32, 
                'beta_slow': 1, 
                'factor': 40, 
                'mscale': 0.707, 
                'mscale_all_dim': 0.707, 
                'original_max_position_embeddings': 4096, 
                'type': 'yarn'
            }
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.ep_size = ep_size
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
