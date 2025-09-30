#!/usr/bin/env python3
"""
Simple test script to check if the compact_rienet package imports correctly.
"""

import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    print("Testing basic imports...")
    
    # Test tensorflow import
    import tensorflow as tf
    print(f"✓ TensorFlow imported successfully: {tf.__version__}")
    
    # Test keras import  
    from keras import layers
    print("✓ Keras imported successfully")
    
    # Test our package structure
    import compact_rienet
    print("✓ compact_rienet package imported successfully")
    
    # Test main components
    from compact_rienet import CompactRIEnetLayer
    print("✓ CompactRIEnetLayer imported successfully")
    
    from compact_rienet import variance_loss_function
    print("✓ variance_loss_function imported successfully")
    
    # Test layer creation
    layer = CompactRIEnetLayer(output_type='weights')
    print("✓ CompactRIEnetLayer created successfully")
    
    # Test with small data
    print("\nTesting with sample data...")
    sample_data = tf.random.normal((2, 5, 10))  # 2 batches, 5 stocks, 10 days
    print(f"Sample data shape: {sample_data.shape}")
    
    # Apply layer
    output = layer(sample_data)
    print(f"✓ Layer forward pass successful, output shape: {output.shape}")
    
    print("\n🎉 All tests passed! The package is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)