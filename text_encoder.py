"""
Text encoding utilities for LLM-based prompt encoding.
Supports extracting and normalizing hidden states from language models.
"""

import torch
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple


def encode_prompt_with_llm(
    tokenizer,
    text_encoder,
    prompts: Union[str, List[str]],
    device: Optional[torch.device] = None,
    max_sequence_length: int = 512,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
    """
    Encode prompts using an LLM with hidden state extraction and normalization.
    
    Args:
        tokenizer: HuggingFace tokenizer
        text_encoder: HuggingFace model with hidden states support
        prompts: Single prompt string or list of prompts
        device: Target device
        max_sequence_length: Maximum sequence length
        use_chat_template: Whether to apply chat template formatting
        enable_thinking: Enable thinking mode for chat template
        
    Returns:
        prompt_embeds: (batch, seq_len, hidden_size) normalized embeddings
        prompt_masks: (batch, seq_len) attention mask (True = attend, False = ignore)
    """
    device = device or next(text_encoder.parameters()).device
    
    if isinstance(prompts, str):
        prompts = [prompts]
    
    # Apply chat template if requested
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        formatted_prompts = []
        for prompt_item in prompts:
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            formatted_prompts.append(prompt_item)
        prompts = formatted_prompts
    
    # Tokenize
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()
    
    # Encode with hidden states
    with torch.no_grad():
        encoder_outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )
    
    # Drop embedding layer and stack hidden states
    # hidden_states[0] is embedding layer, we skip it
    hidden_states = encoder_outputs.hidden_states[1:]
    
    # Stack: (num_layers, batch, seq_len, hidden_size)
    H = torch.stack(hidden_states, dim=0)
    
    # L2 normalize across hidden_size dim
    H_norm = F.normalize(H, p=2, dim=-1)
    
    # Average across layers: (batch, seq_len, hidden_size)
    prompt_embeds = H_norm.mean(dim=0)
    
    return prompt_embeds, prompt_masks

