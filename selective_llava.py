import torch
import copy
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, LlamaTokenizer, AutoImageProcessor
class SelectiveLlavaModel(torch.nn.Module):
    def __init__(self, base_model, skip_layers={'attn': [], 'mlp': []}):
        super().__init__()
        self.model = base_model

        # Access the layers through self.language_model.model.layers
        if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
            self.layers = self.model.language_model.model.layers
        else:
            raise AttributeError("The base model does not have the expected 'language_model.model.layers' structure.")

        num_layers = len(self.layers)
        
        for component, layers in skip_layers.items():
            if not all(0 <= idx < num_layers for idx in layers):
                raise ValueError(f"Invalid layer index in {component} skip list. Must be between 0 and {num_layers-1}")

        print(f"\nComponent skipping pattern:")
        print(f"Attention layers to skip: {sorted(skip_layers['attn'])}")
        print(f"MLP layers to skip: {sorted(skip_layers['mlp'])}")

        for idx, layer in enumerate(self.layers):
            if idx in skip_layers['attn']:
                layer.self_attn = AttentionIdentityModule(layer_idx=idx)
                layer.resid_attn_dropout = ResidualIdentityModule()
            
            if idx in skip_layers['mlp']:
                layer.mlp = MLPIdentityModule()
                layer.resid_mlp_dropout = ResidualIdentityModule()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        kwargs['use_cache'] = True
        return self.model.generate(*args, **kwargs)

class AttentionIdentityModule(torch.nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = 32  # LLaVA 1.5 uses 32 heads
        self.head_dim = 128            # From the debug output we see head_dim is 128

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, **kwargs):
        batch_size = hidden_states.shape[0]
        q_len = hidden_states.shape[1]

        if use_cache:
            if past_key_value is not None:
                # Create new states for the current token
                new_key_state = torch.zeros(
                    (batch_size, self.num_attention_heads, 1, self.head_dim),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
                new_value_state = torch.zeros(
                    (batch_size, self.num_attention_heads, 1, self.head_dim),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )

                # Update the cache using the DynamicCache's update method
                key_state, value_state = past_key_value.update(
                    new_key_state,
                    new_value_state,
                    self.layer_idx,
                    kwargs.get('cache_kwargs', {})
                )
            else:
                # Initial states for the full sequence
                key_state = torch.zeros(
                    (batch_size, self.num_attention_heads, q_len, self.head_dim),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
                value_state = torch.zeros(
                    (batch_size, self.num_attention_heads, q_len, self.head_dim),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )

            return hidden_states, None, (key_state, value_state)
        else:
            return hidden_states, None, None

class MLPIdentityModule(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states

class ResidualIdentityModule(torch.nn.Module):
    def forward(self, hidden_states):
        if isinstance(hidden_states, tuple):
            return hidden_states[0]
        return hidden_states
    
# Initialize model
model_id = "llava-hf/llava-1.5-7b-hf"
base_model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(0)

# Initialize components separately
image_processor = AutoImageProcessor.from_pretrained(model_id)
tokenizer = LlamaTokenizer.from_pretrained(model_id)
processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

# Initialize the SelectiveLlavaModel with some layers to skip
skip_layers = {'attn': [0, 2], 'mlp': [1, 3]}  # Example layer indices to skip
model = SelectiveLlavaModel(base_model, skip_layers=skip_layers)

# Define the prompt directly
prompt = "What are these? <image>"  # Simplified prompt

# Load and process the image
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

# Run the forward pass and generate output
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))