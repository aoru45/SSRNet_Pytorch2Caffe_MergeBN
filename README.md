# Pytorch2Caffe
## Step 1

Build a prototxt (manually).

The name of node in you graph should obey the following rules:

1. If the node has parameters, the name of node should be consistent with your code in pytorch(For example, convolution, batchnorm and innerproduct). The name of nodes without params can be named in any way.
2. If you use batchnorm in batchnorm, you should use BatchNorm and Scale in you caffe graph.

Example:

```python
# pytorch code
class Net(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
        	nn.Conv2d(3,64,3),
            nn.BatchNorm2d(64),
            nn.Tanh()
        )
    def forward(self,x):
        return self.net(x)
```

```json
# caffe graph
...

layer {
  name: "net.0_conv"
  type: "Convolution"
  bottom: "data"
  top: "net0"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

layer {
  name: "net.1_bn"
  type: "BatchNorm"
  bottom: "net0"
  top: "net1_bn"
  batch_norm_param {
    use_global_stats: true
    eps: 9.999999747378752e-06
  }
}
layer {
  name: "net.1_scale"
  type: "Scale"
  bottom: "net1_bn"
  top: "net1_scale"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "tanh1"
  type: "TanH"
  bottom: "net1_scale"
  top: "tanh1"
}
```

## Step 2

Load your pth and prototxt in pycaffe, run the following code for converting the model to caffemodel:

```cmd
python convert.py
```

## Step 3

You should merge the Batchnorm layer and Scale layer in your graph.

just run

```cmd
python merge_bn.py
```

you will get a caffe graph after merging BN.

Note that the output graph will link the output graph automatically so you do not need to link it yourself.

