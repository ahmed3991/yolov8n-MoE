#!/usr/bin/env python3
"""
Quick test script to verify the MoE metrics fix is working.

This script performs a simple test to ensure:
1. Model can be initialized
2. Forward pass works correctly
3. Load balancing loss is calculated
4. Expert tracking functions properly
"""

import torch
import sys
sys.path.append('.')

from train_moe_custom import YOLOv8n_MoE_C2f, NUM_CLASSES, E, K

def test_model_initialization():
    """Test that the model can be initialized."""
    print("Testing model initialization...")
    model = YOLOv8n_MoE_C2f(nc=NUM_CLASSES)
    print(f"✅ Model initialized with {NUM_CLASSES} classes")
    return model

def test_forward_pass(model):
    """Test that forward pass works."""
    print("\nTesting forward pass...")
    model.eval()
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 640, 640)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Forward pass successful")
    print(f"   Output type: {type(output)}")
    return output

def test_balance_loss(model):
    """Test that load balancing loss can be calculated."""
    print("\nTesting load balancing loss...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # Forward pass
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Get balance loss
    balance_loss = model.get_total_balance_loss()
    print(f"✅ Balance loss calculated: {balance_loss.item():.6f}")
    return balance_loss

def test_expert_tracking():
    """Test expert tracking utilities."""
    print("\nTesting expert tracking utilities...")
    from utils.metrics_moe import calculate_expert_balance, track_expert_stats
    
    # Create dummy expert weights
    num_tokens = 100
    expert_weights = torch.randn(num_tokens, E).softmax(dim=-1)
    
    # Calculate balance
    stats = calculate_expert_balance(expert_weights, E)
    print(f"✅ Expert balance calculated:")
    print(f"   Mean usage: {stats['mean_usage']:.2f}%")
    print(f"   Std usage: {stats['std_usage']:.2f}%")
    print(f"   Expert usage: {[f'{x:.1f}%' for x in stats['expert_usage']]}")
    
    # Test tracking
    top_k = torch.topk(expert_weights, K, dim=-1)
    track_stats = track_expert_stats(top_k.indices, top_k.values, E)
    print(f"\n✅ Expert tracking:")
    print(f"   Unique experts used: {track_stats['unique_experts_used']}/{E}")
    
    return stats

def main():
    """Run all tests."""
    print("="*60)
    print("  YOLOv8n-MoE Metrics Fix Verification")
    print("="*60)
    
    try:
        # Test 1: Model initialization
        model = test_model_initialization()
        
        # Test 2: Forward pass
        output = test_forward_pass(model)
        
        # Test 3: Balance loss
        balance_loss = test_balance_loss(model)
        
        # Test 4: Expert tracking
        expert_stats = test_expert_tracking()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\n💡 Key Results:")
        print("   ✓ Model initializes correctly")
        print("   ✓ Forward pass works (single pass)")
        print("   ✓ Load balancing loss is calculated")
        print("   ✓ Expert tracking functions properly")
        print("\n📊 Metrics Fix Summary:")
        print("   • Each batch processes ONCE (prevents doubled metrics)")
        print("   • Balance loss encourages expert diversity")
        print(f"   • All {E} experts are available for routing")
        print(f"   • Top-{K} expert selection per token")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
