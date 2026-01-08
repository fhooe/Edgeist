## Structure of config file
## # Generates a new entry until the next #
## ## are comments and will be ignored like empty lines
## structure of one line: Idx. Name: Amount * Datatype
## idx = will be ignored only of md output
## Name = Name of the Variable
## Amount = optional, how many of the datatype (default = 1)
##          0 is used to define a array/pointer with unknown length
## Datatype = c++ datatype, except for OffsetTable

# Config-Info
- version: 01.00.00
## Add before every Layer
- id: uint8_t
## Size of one entry of every offset_table
- offsetTable: uint32_t

# offsetTable
## structure of the offset_table for each layer with data
## is used in config as OffsetTable datatype
1. pruningMaskOffset: offsetTable
2. weightsMaskOffset: offsetTable
3. weightsTrainableOffset: offsetTable
4. weightsFrozenOffset: offsetTable
5. biasMaskOffset: offsetTable
6. biasTrainableOffset: offsetTable
7. biasFrozenOffset: offsetTable

# Header
1. headerSize: uint16_t
2. magicNumber: uint32_t
3. version: 8 * uint8_t
4. layerNrs: uint16_t
5. channelsIn: uint16_t
6. dimensionInputX: uint32_t
7. dimensionInputY: uint32_t
8. channelsOut: uint16_t
9. dimensionOutputX: uint32_t
10. dimensionOutputY: uint32_t
11. layerOffsetTable: Anzahl_Schichten * uint32_t

# Linear
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. dataEncoding: uint8_t
7. weightsAmountFrozen: uint32_t
8. weightsAmountTrainable: uint32_t
9. biasAmountFrozen: uint16_t
10. biasAmountTrainable: uint32_t
11. offsetTable: OffsetTable
12. predecessors: 0 * uint16_t

# Conv2d
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionInputY: uint32_t
6. dimensionOutputX: uint32_t
7. dimensionOutputY: uint32_t
8. channelsIn: uint16_t
9. channelsOut: uint16_t
10. kernelSize: 2 * uint8_t
11. padding: 2 * uint8_t
12. stride: 2 * uint8_t
13. dilation: 2 * uint8_t
14. groups: uint8_t
15. dataEncoding: uint8_t
16. weightsAmountFrozen: uint32_t
17. weightsAmountTrainable: uint32_t
18. biasAmountFrozen: uint16_t
19. biasAmountTrainable: uint32_t
20. offsetTable: OffsetTable
21. predecessors: 0 * uint16_t

# MaxPool2d
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionInputY: uint32_t
6. dimensionOutputX: uint32_t
7. dimensionOutputY: uint32_t
8. channelsIn: uint16_t
9.  channelsOut: uint16_t
10. kernelSize: uint8_t
11. padding: uint8_t
12. stride: uint8_t
13. dilation: uint8_t
14. predecessors: 0 * uint16_t

# ReLU
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. predecessors: 0 * uint16_t

# Softmax
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. predecessors: 0 * uint16_t

# Flatten
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. predecessors: 0 * uint16_t

# BatchNorm1d
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. dataEncoding: uint8_t
7. weightsAmountFrozen: uint32_t
8. weightsAmountTrainable: uint32_t
9. biasAmountFrozen: uint16_t
10. biasAmountTrainable: uint32_t
11. offsetTable: OffsetTable
12. predecessors: 0 * uint16_t

# BatchNorm2d
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. channelsIn: uint16_t
7. channelsOut: uint16_t
8. dataEncoding: uint8_t
9. weightsAmountFrozen: uint32_t
10. weightsAmountTrainable: uint32_t
11. biasAmountFrozen: uint16_t
12. biasAmountTrainable: uint32_t
13. offsetTable: OffsetTable
14. predecessors: 0 * uint16_t

# AdaptiveAvgPool1d
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. channelsIn: uint16_t
7. channelsOut: uint16_t
8.  predecessors: 0 * uint16_t

# AdaptiveAvgPool2d
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionInputY: uint32_t
6. dimensionOutputX: uint32_t
7. dimensionOutputY: uint32_t
8. channelsIn: uint16_t
9. channelsOut: uint16_t
10. predecessors: 0 * uint16_t

# Dropout
2. layerNr: uint8_t
3. predecessorNr: uint8_t
4. dimensionInputX: uint32_t
5. dimensionOutputX: uint32_t
6. DropoutRate: float32_t
7. predecessors: 0 * uint16_t