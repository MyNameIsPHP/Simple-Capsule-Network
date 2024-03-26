# Simple-Capsule-Network
Simple implementation of CapsNet in Tensorflow 2.0
## Usage:
```bash
  primary_caps = PrimaryCaps(num_capsules=32, dim_capsule=8, kernel_size=9, strides=2, padding='valid')(x)

  # Following with a Capsule Layer
  digit_caps = CapsuleLayer(num_capsules=10, dim_capsule=16, routings=3)(primary_caps)
```
