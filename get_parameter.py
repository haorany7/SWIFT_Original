import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def print_model_params(model_id="microsoft/Phi-3.5-mini-instruct"):
    """Print relevant model parameters for KV cache size calculation"""
    print(f"\nAnalyzing model: {model_id}")
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto",
                                                 low_cpu_mem_usage=True)
    config = model.config
    
    # Print basic model info
    print("\nModel Configuration:")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Head dimension: {config.hidden_size // config.num_attention_heads}")
    #print the max_position_embeddings
    print(f"Max position embeddings: {config.max_position_embeddings}")
    # Calculate KV cache size
    batch_size = 1
    max_length = 2048
    bytes_per_element = 2  # float16
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Calculate sizes
    size_per_layer = (
        batch_size * 
        config.num_attention_heads * 
        max_length * 
        head_dim * 
        2 *  # for keys and values
        bytes_per_element
    )
    
    total_size = size_per_layer * config.num_hidden_layers
    
    # Print memory requirements
    print("\nMemory Requirements:")
    print(f"Size per layer: {size_per_layer / (1024**3):.2f} GB")
    print(f"Total KV cache size: {total_size / (1024**3):.2f} GB")
    
    # Print GPU info if available
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"\nGPU Memory:")
        print(f"Total GPU memory: {gpu_mem / (1024**3):.2f} GB")
        print(f"Maximum safe KV cache length: {int((gpu_mem * 0.8) / size_per_layer)}")

if __name__ == "__main__":
    # Print memory status before loading
    if torch.cuda.is_available():
        print("Initial GPU Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    # Check model parameters
    print_model_params()
    
    # Print final memory status
    if torch.cuda.is_available():
        print("\nFinal GPU Memory Status:")
        print(f"Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")