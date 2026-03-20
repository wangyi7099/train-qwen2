from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.processing_utils import Unpack
from transformers.integrations import use_kernel_forward_from_hub
import torch
from torch import nn
import matplotlib
from transformers import PreTrainedModel, GenerationMixin
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from typing import Callable, Optional, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
matplotlib.use('Agg')  # 设置非交互式后端


class DecoderOnlyModelConfig(PretrainedConfig):
    def __init__(self, 
                 vocab_size = 151936, 
                 num_hiddens = 896, 
                 num_heads = 14, 
                 num_layers = 24,  
                 rms_norm_eps = 1e-06, 
                 max_position_embeddings=32768, num_key_value_heads=32, sliding_window=4096, layer_types=None, max_window_layers=28,  attention_dropout=0.0, **kwargs):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_key_value_heads = num_key_value_heads
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.max_window_layers = max_window_layers

        if self.layer_types is None:
            self.layer_types = [
                'sliding_attention'
                if self.sliding_window is not None and i >= self.max_window_layers
                else 'full_attention'
                for i in range(self.num_layers)
            ]
        super().__init__(
            **kwargs,
        )
        pass


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class PositionWiseFFN(nn.Module):

    def __init__(self, config: DecoderOnlyModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.num_hiddens
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, X):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(X)) * self.up_proj(X))

        return down_proj


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        casual_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + casual_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    def __init__(self, config: DecoderOnlyModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, 'head_dim', config.num_hiddens // config.num_heads)
        self.num_key_value_groups = config.num_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.num_hiddens, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(
            config.num_hiddens, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(
            config.num_hiddens, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            config.num_heads * self.head_dim, config.num_hiddens, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[
            layer_idx] == 'sliding_attention' else None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface = eager_attention_forward
        # if self.config._attn_implementation != 'eager':
        #     attention_interface =
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class DecoderBlock(nn.Module):
    """解码器第i个块"""

    def __init__(self, config: DecoderOnlyModelConfig,  i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = Qwen2Attention(config=config, layer_idx=i)
        self.addnorm1 = Qwen2RMSNorm(config.num_hiddens, config.rms_norm_eps)
        self.addnorm2 = Qwen2RMSNorm(config.num_hiddens, config.rms_norm_eps)
        self.ffn = PositionWiseFFN(config=config)
        self.attention_type = config.layer_types[i]

    def forward(self, hidden_states, attention_mask, position_ids, past_key_values, use_cache, cache_position, position_embeddings, **kwargs):
        residual = hidden_states
        hidden_states = self.addnorm1(hidden_states)
        hidden_states, _ = self.attention1(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        residual = residual + hidden_states

        residual = hidden_states

        hidden_states = self.addnorm2(hidden_states)
        hidden_states = self.ffn(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


def _compute_default_rope_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional["torch.device"] = None,
    seq_len: Optional[int] = None,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(
        config, "partial_rotary_factor") else 1.0
    head_dim = getattr(
        config, "head_dim", None) or config.num_hiddens // config.num_heads
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2,
                      dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: DecoderOnlyModelConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    # power user: used with advanced RoPE types (e.g. dynamic rope)
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_preq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        ).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_preq_expanded.float() @
                     position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DecoderModel(PreTrainedModel):
    def __init__(self, config: DecoderOnlyModelConfig):
        super().__init__(config)
        self.num_hiddens = config.num_hiddens
        self.num_layers = config.num_layers
        self.embedding = nn.Embedding(config.vocab_size, config.num_hiddens)
        self.blks = nn.Sequential()
        self.norm = Qwen2RMSNorm(config.num_hiddens, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        for i in range(self.num_layers):
            self.blks.add_module("block"+str(i),
                                 DecoderBlock(config, i))
        self.dense = nn.Linear(config.num_hiddens, config.vocab_size)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length(
            ) if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                    **mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.blks[:self.num_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)


class DecoderOnlyModelDecoder(PreTrainedModel, GenerationMixin):
    # def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
    #              num_heads, num_layers, dropout, **kwargs):
    supports_gradient_checkpointing = True

    def __init__(self, config: DecoderOnlyModelConfig):
        super().__init__(config)
        self.model = DecoderModel(config)
        self.lm_head = nn.Linear(
            config.num_hiddens, config.vocab_size, bias=False)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep,
                              None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
