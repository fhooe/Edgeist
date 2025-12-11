## Structure of config file
## # Generates a new entry until the next #
## ## are comments and will be ignored like empty lines
## structure of one line: Idx. Name: Amount * Datatype
## idx = will be ignored only of md output
## Name = Name of the Variable
## Amount = optional, how many of the datatype (default = 1)
##          0 is used to define a array/pointer with unknown length
## Datatype = c++ datatype, excepte for Offset_Table_t

# Config-Info
- Version: 01.00.00
## Add befor every Layer
- ID: uint8_t
## Size of one entry of every offset_table
- Offset_Table: uint32_t

# Offset_Table
## structure of the offset_table for each layer with data
## is used in config as Offset_Table_t datatype
1. Pruning_mask_offset: Offset_Table
2. Weights_mask_offset: Offset_Table
3. Weights_trainable_offset: Offset_Table
4. Weights_frozen_offset: Offset_Table
5. Bias_mask_offset: Offset_Table
6. Bias_trainable_offset: Offset_Table
7. Bias_frozen_offset: Offset_Table

# Header
1. Header_Size: uint16_t
2. Magic_Number: uint32_t
3. Version: 8 * uint8_t
4. LayerNrs: uint16_t
5. ChannelsIn: uint16_t
6. DimensionInput_x: uint32_t
7. DimensionInput_y: uint32_t
8. ChannelsOut: uint16_t
9. DimensionOutput_x: uint32_t
10. DimensionOutput_y: uint32_t
11. Layer_Offset_Table: Anzahl_Schichten * uint32_t

# Linear
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. Dataencoding: uint8_t
7. Weights_amount_frozen: uint32_t
8. Weights_amount_trainable: uint32_t
9. Bias_amount_frozen: uint16_t
10. Bias_amount_trainable: uint32_t
11. Offset_Table: Offset_Table_t
12. predecessors: 0 * uint16_t

# Conv2d
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionInput_y: uint32_t
6. DimensionOutput_x: uint32_t
7. DimensionOutput_y: uint32_t
8. ChannelsIn: uint16_t
9. ChannelsOut: uint16_t
10. KernelSize: 2 * uint8_t
11. padding: 2 * uint8_t
12. stride: 2 * uint8_t
13. dilation: 2 * uint8_t
14. groups: uint8_t
15. Dataencoding: uint8_t
16. Weights_amount_frozen: uint32_t
17. Weights_amount_trainable: uint32_t
18. Bias_amount_frozen: uint16_t
19. Bias_amount_trainable: uint32_t
20. Offset_Table: Offset_Table_t
21. predecessors: 0 * uint16_t

# MaxPool2d
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionInput_y: uint32_t
6. DimensionOutput_x: uint32_t
7. DimensionOutput_y: uint32_t
8. ChannelsIn: uint16_t
9.  ChannelsOut: uint16_t
10. KernelSize: uint8_t
11. padding: uint8_t
12. stride: uint8_t
13. dilation: uint8_t
14. predecessors: 0 * uint16_t

# ReLU
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. predecessors: 0 * uint16_t

# Softmax
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. predecessors: 0 * uint16_t

# Flatten
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. predecessors: 0 * uint16_t

# BatchNorm1d
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. Dataencoding: uint8_t
7. Weights_amount_frozen: uint32_t
8. Weights_amount_trainable: uint32_t
9. Bias_amount_frozen: uint16_t
10. Bias_amount_trainable: uint32_t
11. Offset_Table: Offset_Table_t
12. predecessors: 0 * uint16_t

# BatchNorm2d
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. ChannelsIn: uint16_t
7. ChannelsOut: uint16_t
8. Dataencoding: uint8_t
9. Weights_amount_frozen: uint32_t
10. Weights_amount_trainable: uint32_t
11. Bias_amount_frozen: uint16_t
12. Bias_amount_trainable: uint32_t
13. Offset_Table: Offset_Table_t
14. predecessors: 0 * uint16_t

# AdaptiveAvgPool1d
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. ChannelsIn: uint16_t
7. ChannelsOut: uint16_t
8.  predecessors: 0 * uint16_t

# AdaptiveAvgPool2d
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionInput_y: uint32_t
6. DimensionOutput_x: uint32_t
7. DimensionOutput_y: uint32_t
8. ChannelsIn: uint16_t
9. ChannelsOut: uint16_t
10. predecessors: 0 * uint16_t

# Dropout
2. LayerNr: uint8_t
3. predecessorNr: uint8_t
4. DimensionInput_x: uint32_t
5. DimensionOutput_x: uint32_t
6. DropoutRate: float32_t
7. predecessors: 0 * uint16_t