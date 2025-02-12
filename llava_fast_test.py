from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
import torch.nn.functional as F
import copy
import time
from PIL import Image

class SelectiveLlavaModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = copy.deepcopy(base_model)
        
        # Pre-defined optimal skip layers
        skip_layers = {
            'attn': [5, 9, 12, 13, 14, 20, 21, 22, 23, 24, 29, 30],
            'mlp': [6, 7, 10, 11, 12, 17, 18, 21, 22, 25, 26, 27, 28, 29, 30]
        }
        
        print(f"\nSkipping layers:")
        print(f"Attention layers: {skip_layers['attn']}")
        print(f"MLP layers: {skip_layers['mlp']}")

        # Replace layers with identity modules
        for idx, layer in enumerate(self.model.language_model.model.layers):
            if idx in skip_layers['attn']:
                layer.self_attn = AttentionIdentityModule()
                if hasattr(layer, 'post_attention_layernorm'):
                    layer.post_attention_layernorm = ResidualIdentityModule()
            
            if idx in skip_layers['mlp']:
                layer.mlp = MLPIdentityModule()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs['use_cache'] = True
        return self.model.generate(*args, **kwargs)

class AttentionIdentityModule(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        outputs = (hidden_states,)
        if kwargs.get('output_attentions', False):
            outputs += (None,)
        if kwargs.get('use_cache', False):
            batch_size = hidden_states.size(0)
            head_dim = hidden_states.size(-1)
            dummy_state = torch.zeros((batch_size, 1, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
            outputs += ((dummy_state, dummy_state),)
        return outputs

class MLPIdentityModule(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states

class ResidualIdentityModule(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states if not isinstance(hidden_states, tuple) else hidden_states[0]

class SpeculativeDecoder:
    def __init__(self, base_model, draft_model, processor, num_guesses=5):
        self.base_model = base_model
        self.draft_model = draft_model
        self.processor = processor
        self.num_guesses = num_guesses

    def safe_sample_token(self, logits, temperature=1.0, top_k=40, top_p=0.85):
        if len(logits.shape) == 3:
            logits = logits.squeeze(1)
            
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        if top_k > 0:
            values, _ = torch.topk(probs, min(top_k, probs.size(-1)))
            probs = torch.where(probs < values[:, -1].unsqueeze(-1), torch.zeros_like(probs), probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        if 0.0 < top_p < 1.0:
            sorted_probs, indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum <= top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = True
            probs = torch.zeros_like(probs).scatter_(-1, indices, sorted_probs * mask)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)

    def speculative_decode(self, inputs, max_new_tokens=100, temperature=1.0, top_k=25, top_p=0.85):
        input_ids = inputs['input_ids'].clone()
        initial_length = input_ids.shape[1]
        start_time = time.time()
        
        while input_ids.shape[1] < max_new_tokens + initial_length:
            with torch.no_grad():
                # Draft phase
                draft_outputs = self.draft_model.generate(
                    **inputs,
                    max_new_tokens=self.num_guesses,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                draft_tokens = draft_outputs.sequences[0, input_ids.shape[1]:].tolist()
                if not draft_tokens:
                    continue

                # Verification phase
                full_sequence = torch.cat([
                    input_ids,
                    torch.tensor([draft_tokens], device=input_ids.device)
                ], dim=1)
                
                draft_verify = self.draft_model(
                    **{**inputs, 'input_ids': full_sequence}
                )
                base_verify = self.base_model(
                    **{**inputs, 'input_ids': full_sequence}
                )
                
                # Process logits and calculate acceptance
                draft_logits = draft_verify.logits[:, input_ids.shape[1]-1:-1, :]
                base_logits = base_verify.logits[:, input_ids.shape[1]-1:-1, :]
                
                draft_probs = F.softmax(draft_logits / temperature, dim=-1)
                base_probs = F.softmax(base_logits / temperature, dim=-1)
                
                token_indices = torch.tensor(draft_tokens, device=input_ids.device).unsqueeze(0).unsqueeze(-1)
                draft_token_probs = torch.gather(draft_probs, 2, token_indices).squeeze(-1)
                base_token_probs = torch.gather(base_probs, 2, token_indices).squeeze(-1)
                
                acceptance_ratios = (base_token_probs / (draft_token_probs + 1e-8)).clamp(0, 1)
                accepted_mask = torch.rand_like(acceptance_ratios) <= acceptance_ratios
                
                n_accepted = accepted_mask[0].int().sum().item()
                
                if n_accepted > 0:
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([draft_tokens[:n_accepted]], device=input_ids.device)
                    ], dim=1)
                
                if n_accepted < len(draft_tokens):
                    logits = base_verify.logits[:, input_ids.shape[1]-1:input_ids.shape[1], :]
                    token = self.safe_sample_token(logits, temperature, top_k, top_p)
                    input_ids = torch.cat([input_ids, token], dim=1)
                
                if input_ids[0, -1].item() == self.processor.tokenizer.eos_token_id:
                    break

        tokens_generated = input_ids.shape[1] - initial_length
        generation_time = time.time() - start_time
        
        return input_ids, tokens_generated, generation_time
# Cell 6: Initialize models and decoder
def initialize_models():
    print("Loading LLaVA model and processor...")
    model_id = "llava-hf/llava-1.5-13b-hf"
    
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.float16,
        use_flash_attention_2=False
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    print("Creating draft model...")
    draft_model = SelectiveLlavaModel(base_model)
    
    print("Initializing speculative decoder...")
    decoder = SpeculativeDecoder(
        base_model=base_model,
        draft_model=draft_model,
        processor=processor,
        num_guesses=5
    )
    
    return decoder, processor

# Cell 7: Test the implementation
def test_generation(decoder, processor, image_path, prompt="Describe this image"):
    # Load and process image
    image = Image.open(image_path)
    
    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to("cuda")
    
    print("Generating with speculative decoding...")
    output_ids, tokens_generated, generation_time = decoder.speculative_decode(
        inputs=inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=25,
        top_p=0.85
    )
    
    # Decode output
    output_text = processor.decode(
        output_ids[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    print(f"\nGeneration Results:")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens per second: {tokens_generated/generation_time:.2f}")
    print(f"\nOutput text:\n{output_text}")
    
    return output_text
def main():
        # Cell 8: Run the test with URL image
    import requests

    # Download image from URL
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # Initialize models and run test
    decoder, processor = initialize_models()

    # Process inputs
    inputs = processor(
        images=image,
        text="Describe this image in detail",
        return_tensors="pt"
    ).to("cuda")

    print("Generating with speculative decoding...")
    output_ids, tokens_generated, generation_time = decoder.speculative_decode(
        inputs=inputs,
        max_new_tokens=200,  # Increased for more detailed description
        temperature=0.7,
        top_k=25,
        top_p=0.85
    )

    # Decode output
    output_text = processor.decode(
        output_ids[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print(f"\nGeneration Results:")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Tokens per second: {tokens_generated/generation_time:.2f}")
    print(f"\nOutput text:\n{output_text}")

    # Optional: Standard generation for comparison
    print("\nGenerating with standard method for comparison...")
    with torch.no_grad():
        start_time = time.time()
        standard_output = decoder.base_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=25,
            top_p=0.85,
            do_sample=True
        )
        standard_time = time.time() - start_time

    standard_text = processor.decode(
        standard_output[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print(f"\nStandard Generation Results:")
    print(f"Generation time: {standard_time:.2f}s")
    print(f"\nStandard output text:\n{standard_text}")

    # Calculate speedup
    speedup = standard_time / generation_time
    print(f"\nSpeedup: {speedup:.2f}x")
if __name__ == "__main__":
    main()