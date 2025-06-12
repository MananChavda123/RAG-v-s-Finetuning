#!/usr/bin/env python3
"""
Script to load DistilBERT model from Hugging Face
Supports both tokenizer and model loading with various configurations
"""

from transformers import (
    DistilBertTokenizer, 
    DistilBertModel, 
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
import torch

def load_distilbert_base():
    """Load the base DistilBERT model and tokenizer"""
    print("Loading DistilBERT base model...")
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    print(f"Model loaded: {model.config.model_type}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    return tokenizer, model

def load_distilbert_classification():
    """Load DistilBERT for sequence classification"""
    print("Loading DistilBERT for sequence classification...")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2  # Binary classification
    )
    
    return tokenizer, model

def load_with_auto_classes():
    """Load using Auto classes (recommended approach)"""
    print("Loading with Auto classes...")
    
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    return tokenizer, model

def test_model(tokenizer, model, text="Hello, this is a test sentence."):
    """Test the loaded model with sample text"""
    print(f"\nTesting with text: '{text}'")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print output shape
    if hasattr(outputs, 'last_hidden_state'):
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        print(f"Hidden states for [CLS] token: {outputs.last_hidden_state[0, 0, :5]}")
    else:
        print(f"Logits shape: {outputs.logits.shape}")

def load_with_custom_config():
    """Load model with custom configuration"""
    print("Loading with custom configuration...")
    
    from transformers import DistilBertConfig
    
    # Create custom config
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    config.output_hidden_states = True
    config.output_attentions = True
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
    
    return tokenizer, model

def main():
    """Main function to demonstrate different loading methods"""
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # # Method 1: Load base model
        # print("=" * 50)
        # tokenizer1, model1 = load_distilbert_base()
        # model1.to(device)
        # test_model(tokenizer1, model1)
        
        # Method 2: Load for classification
        print("\n" + "=" * 50)
        tokenizer2, model2 = load_distilbert_classification()
        model2.to(device)
        test_model(tokenizer2, model2)
        
        # # Method 3: Load with Auto classes
        # print("\n" + "=" * 50)
        # tokenizer3, model3 = load_with_auto_classes()
        # model3.to(device)
        # test_model(tokenizer3, model3)
        
        # # Method 4: Load with custom config
        # print("\n" + "=" * 50)
        # tokenizer4, model4 = load_with_custom_config()
        # model4.to(device)
        
        # Test with attention and hidden states
        inputs = tokenizer2("Test sentence", return_tensors='pt')
        with torch.no_grad():
            outputs = model2(**inputs.to(device))
        
        print(f"Hidden states layers: {len(outputs.hidden_states)}")
        print(f"Attention layers: {len(outputs.attentions)}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have transformers installed: pip install transformers torch")

if __name__ == "__main__":
    main()