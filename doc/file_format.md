# File format for model translation

## Idea

In order to use a neural network with a µC in both the forward pass and the backward pass, the model generated (with pytorch) should be saved in a format that enables efficient processing on a µC.

## Data format

- Endianness
- Alignment

## Format

The format consists of the following elements:

1. Header
   1. Header size
   2. Magic number
      1. 0x4E4D434D (NMCF) (Neural Micro Controller Framework)
   3. Basic information about the network
      1. version (not all layer types and activation functions are supported in version 1)
      2. Feature map: what was used in this network (data types, supported layers, activation functions, etc.)
      3. File size
      4. Number of layers (layers and activation functions are separate layers)
      5. Input format
         1. Size
      6. Output format
         1. Size
   4. Offset table
      1. Offset for efficient addressing of the respective layers
      2. Layers (for details on layer types, see the section below)
      3. Backup
         1. Checksum

## Layers

### Fully Connected

1. Layer No.: int
2. ID: 1
3. predecessorNo.: int
4. predecessors[predecessorNo.]: int
5. Structure
   1. dimensionInputX: int
   2. dimensionOutputX: int
   3. Data encoding: Datatype
   4. pruned: bool
   5. trainableWeights: int
   6. trainableBias: int
6. Data:
   1. Data Header
      1. Offset pruneMask
      2. Offset weightsMask
      3. Offset weightsTrainable
      4. Offset weightsFrozen
      5. Offset biasMask
      6. Offset biasTrainable
      7. Offset biasFrozen
   2. if (pruned) Prune_mask [kernelSize[0] * kernelSize[1] * channelsOut + dimensionOutputX * dimensionOutputY]: bool
   3. if (trainableWeights>0) weightsMask [kernelSize[0] * kernelSize[1] * channelsOut + dimensionOutputX * dimensionOutputY]: bool
   4. weightsTrainable[ trainableWeights ]: Datatype //loaded into SRAM, empty if all are frozen
   5. weightsFrozen[size * size * channelsOut - trainableWeights]: Datatype //remains in flash, is empty if all are trainable
   6. if (trainableWeights>0) biasMask [kernelSize[0] * kernelSize[1] * channelsOut + dimensionOutputX * dimensionOutputY]: bool
   7. biasTrainable[ trainableBias]: Datatype //loaded into SRAM, empty if all are frozen
   8. biasFrozen[channelsOut - trainableBias]: Datatype //remains in flash, empty if all are trainable

#### Explanation

Depending on the bitmask entry of trainableMask and pruneMask, the respective pointers in weightsTrainable or weightsFrozen are increased or not.
+ Test case

### Conv1D

1. Layer No.: int
2. ID: 2
3. predecessorNo.: int
4. predecessors[predecessorNo.]: int
5. Dimension
   1. dimensionInputX: int
   2. dimensionOutputX: int
   3. channelsIn: int
   4. channelsOut: int
   5. kernelSize: [int,int]
   6. padding: [int,int]
   7. stride: [int,int]
   8. dilation: [int,int]
   9. groups: int
   10. Data encoding: Datatype
   11. pruned: bool
   12. trainableWeights: int
   13. trainableBias: int
6. Data
   1. Data Header
      1. Offset trainableMask
      2. Offset pruneMask
      3. Offset Kernel_trainable
      4. Offset Kernel_frozen
      5. Offset biasTrainable
      6. Offset biasFrozen
   2. if (trainableWeights>0 || trainableBias >0) trainableMask [(kernelSize + dimensionOutputX]: bool
   3. if (pruned) pruneMask [(kernelSize + dimensionOutputX]: bool
   4. weightsTrainable[ trainableWeights ]: Datatype //loaded into SRAM, empty if all are frozen
   5. weightsFrozen[kernelSize * channelsOut - trainableWeights]: Datatype //remains in flash, is empty if all are trainable
   6. biasTrainable[ trainableBias]: Datatype //loaded into SRAM, is empty if all are frozen
   7. biasFrozen[channelsOut - trainableBias]: Datatype //remains in flash, is empty if all are trainable

### Conv2D

1. Layer No.: int
2. ID: 3
3. predecessorNo.: int
4. predecessors[predecessorNo.]: int
5. Dimension
   1. dimensionInputX: int
   2. dimensionInputY: int
   3. dimensionOutputX: int
   4. dimensionOutputX: int
   5. channelsIn: int
   6. channelsOut: int
   7. kernelSize: [int,int]
   8. padding: [int,int]
   9. stride: [int,int]
   10. dilation: [int,int]
   11. groups: int
   12. Data encoding: Datatype (enum, float32, float 16, .....)
   13. pruned: bool
   14. trainableWeights: int
   15. trainableBias: int
6. Data
   1. Data Header
      1. Offset trainableMask
      2. Offset pruneMask
      3. Offset Kernel_trainable
      4. Offset Kernel_frozen
      5. Offset biasTrainable
      6. Offset biasFrozen
   2. if (pruned) Prune_mask [kernelSize[0] * kernelSize[1] * channelsOut + dimensionOutputX * dimensionOutputY]: bool
   3. if (trainableWeights>0) weightsMask [kernelSize[0] * kernelSize[1] * channelsOut + dimensionOutputX * dimensionOutputY]: bool
   4. weightsTrainable[ trainableWeights ]: Datatype //loaded into SRAM, empty if all are frozen
   5. weightsFrozen[size * size * channelsOut - trainableWeights]: Datatype //remains in flash, is empty if all are trainable
   6. if (trainableWeights>0) biasMask [kernelSize[0] * kernelSize[1] * channelsOut + dimensionOutputX * dimensionOutputY]: bool
   7. biasTrainable[ trainableBias]: Datatype //loaded into SRAM, empty when all are frozen
   8. biasFrozen[channelsOut - trainableBias]: Datatype //remains in flash, empty when all are trainable

### Depth-wise Convolution

= conv2D with groups = channelsIn

### MaxPool2d

1. Layer No.: int
2. ID: 5
3. predecessorNo.: int
4. predecessors[predecessorNo.]: int
5. Dimension
   1. dimensionInputX: int
   2. dimensionInputY: int
   3. dimensionOutputX: int
   4. dimensionOutputY: int
   5. channelsIn: int
   6. channelsIn: out
   7. kernelSize: [int,int]
   8. padding: [int,int]
   9. stride: [int,int]
   10. dilation: [int,int]
6. Data
   1. none

### ReLU

1. Layer No.: int
2. ID: 6
3. predecessorNo.: int
4. predecessors[predecessorNo.]: int
5. Dimension
   1. dimensionInputX: int
   2. dimensionOutputX: int
6. Data
   1. none

### Other layers will be added during the course of the work.

## Open issues/questions

- Parameters for optimizer in the layers?
- Packing
- and
- unpacking
- Predecessor in addition to layer number
- If there are multiple predecessors, how are they arranged (if PyTorch doesn't like it, screw it)
  - Weights are simply added element by element
- Trainable weights (bit mask for frozen_weights in pytorch) + layer by layer

## Next Steps

1. Class diagram
2. Sequence diagram for retraining run
3. Initial implementation
4. Next class diagram

## V1 known and accepted limitations

- 2DConvolutions
  - Kernel_size, stride, padding, and dilation only supported with x = y. The data format supports parameters with different height-to-weight ratios, but the implementation does not.
