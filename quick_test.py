#!/usr/bin/env python3
"""
Quick test to verify the installed package works correctly.
"""

try:
    import compact_rienet
    print("Package imported successfully!")
    
    layer = compact_rienet.CompactRIEnetLayer()
    print("Layer created successfully!")
    
    # Test with small data
    import tensorflow as tf
    sample_data = tf.random.normal((2, 3, 5))
    output = layer(sample_data)
    print(f"Layer test successful! Output shape: {output.shape}")
    
    print("All tests passed - package is working correctly!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()