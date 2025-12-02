# File Format für Model übersetzung

## Idee

Um ein neurales Netz mit einem µC sowohl im Forward pass als auch im Backward pass verwenden zu könne, soll das (mit pytorch) generierte Modell in einem Format gespeichert werden, das eine effiziente verarbeitung auf einem µC ermöglicht.

## Data Format

- Endianness
- Alignment

## Format

Das Format besteht aus folgenden Elementen:

1. Header
   1. Header Size
   2. Magic number
      1. 0x4E4D434D (NMCF) (Neural Micro Controller Framework)
   3. Grundinformationen über das Netzwerk
      1. Version (nicht alle Schichttypen und Aktivierungsfunktionen werden in Version 1 unterstützt)
      2. Featuremap: was wurde in diesem Netz Verwendet(Datentypen, Unterstützte Layer, Aktivierungsfunktionen, ...)
      3. Filesize
      4. Anzahl der Schichten (Layer und Aktivierungsfunktione sind jeweils eigene Schichten)
      5. Input Format
         1. Size
      6. Output Format
         1. Size
   4. Offset Tabelle
      1. Offset um die jeweiligen Schichten effizient addressiern zu können
2. Schichten (Detail zu Schichttyp siehe einen Punkt weiter unten)
3. Sicherung
   1. Checksum

## Schichten

### Fully Connected

1. Schicht Nr: int
2. ID: 1
3. predecessorNr: int
4. predecessors[predecessorNr]: int
5. Structure
   1. DimensionImput_x: int
   2. DimensionOutput_x: int
   3. Data encoding: Datatype
   4. pruned: bool
   5. trainableWeights: int
   6. trainableBias: int
6. Data:
   1. Data Header
      1. Offset pruneMask
      2. Offset Weights_mask
      3. Offset Weights_trainable
      4. Offset Weights_frozen
      5. Offset Bias_mask
      6. Offset Bias_trainable
      7. Offset Bias_frozen
   2. if (pruned) Prune_mask [KernelSize[0] * KernelSize[1] * ChannelsOut + DimensionOutput_x * DimensionOutput_y]: bool
   3. if (trainableWeights>0) Weights_mask [KernelSize[0] * KernelSize[1] * ChannelsOut + DimensionOutput_x * DimensionOutput_y]: bool
   4. Weights_trainable[ trainableWeights ]: Datatype //wird in den Sram geladen, ist leer wenn alle frozen sind
   5. Weights_frozen[size * size * ChannelsOut - trainableWeignts]: Datatype //bleibt im flash, ist leer, wenn alles trainable sind
   6. if (trainableWeights>0) Bias_mask [KernelSize[0] * KernelSize[1] * ChannelsOut + DimensionOutput_x * DimensionOutput_y]: bool
   7. Bias_trainable[ trainableBias]: Datatype //wird in den Sram geladen, ist leer wenn alle frozen sind
   8. Bias_frozen[ChannelsOut - trainableBias]: Datatype //bleibt im flash, ist leer, wenn alles trainable sind

#### Erklärung

Je nach bitmasken eintrag von trainableMask und pruneMask werden die jeweiligen pointer in Weights_trainable oder Weigts_frozen erhöht, oder nicht
+ Testcase

### Conv1D

1. Schicht Nr: int
2. ID: 2
3. predecessorNr: int
4. predecessors[predecessorNr]: int
5. Dimension
   1. DimensionInput_x: int
   2. DimensionOutput_x: int
   3. ChannelsIn: int
   4. ChannelsOut: int
   5. KernelSize: [int,int]
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
      5. Offset Bias_trainable
      6. Offset Bias_frozen
   2. if (trainableWeights>0 || trainableBias >0) trainableMask [(KernelSize + DimensionOutput_x]: bool
   3. if (pruned) pruneMask [(KernelSize + DimensionOutput_x]: bool
   4. Weights_trainable[ trainableWeights ]: Datatype //wird in den Sram geladen, ist leer wenn alle frozen sind
   5. Weights_frozen[KernelSize * ChannelsOut - trainableWeights]: Datatype //bleibt im flash, ist leer, wenn alles trainable sind
   6. Bias_trainable[ trainableBias]: Datatype //wird in den Sram geladen, ist leer wenn alle frozen sind
   7. Bias_frozen[ChannelsOut - trainableBias]: Datatype //bleibt im flash, ist leer, wenn alles trainable sind

### Conv2D

1. Schicht Nr: int
2. ID: 3
3. predecessorNr: int
4. predecessors[predecessorNr]: int
5. Dimension
   1. DimensionInput_x: int
   2. DimensionInput_y: int
   3. DimensionOutput_x: int
   4. DimensionOutput_x: int
   5. ChannelsIn: int
   6. ChannelsOut: int
   7. KernelSize: [int,int]
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
      5. Offset Bias_trainable
      6. Offset Bias_frozen
   2. if (pruned) Prune_mask [KernelSize[0] * KernelSize[1] * ChannelsOut + DimensionOutput_x * DimensionOutput_y]: bool
   3. if (trainableWeights>0) Weights_mask [KernelSize[0] * KernelSize[1] * ChannelsOut + DimensionOutput_x * DimensionOutput_y]: bool
   4. Weights_trainable[ trainableWeights ]: Datatype //wird in den Sram geladen, ist leer wenn alle frozen sind
   5. Weights_frozen[size * size * ChannelsOut - trainableWeignts]: Datatype //bleibt im flash, ist leer, wenn alles trainable sind
   6. if (trainableWeights>0) Bias_mask [KernelSize[0] * KernelSize[1] * ChannelsOut + DimensionOutput_x * DimensionOutput_y]: bool
   7. Bias_trainable[ trainableBias]: Datatype //wird in den Sram geladen, ist leer wenn alle frozen sind
   8. Bias_frozen[ChannelsOut - trainableBias]: Datatype //bleibt im flash, ist leer, wenn alles trainable sind

### Depth-wise Convolution

= conv2D mit groups = ChannelsIn

### MaxPool2d

1. Schicht Nr: int
2. ID: 5
3. predecessorNr: int
4. predecessors[predecessorNr]: int
5. Dimension
   1. DimensionInput_x: int
   2. DimensionInput_y: int
   3. DimensionOutput_x: int
   4. DimensionOutput_y: int
   5. ChannelsIn: int
   6. ChannelsIn: out
   7. KernelSize: [int,int]
   8. padding: [int,int]
   9. stride: [int,int]
   10. dilation: [int,int]
6. Data
   1. none

### ReLU

1. Schicht Nr: int
2. ID: 6
3. predecessorNr: int
4. predecessors[predecessorNr]: int
5. Dimension
   1. DimensionInput_x: int
   2. DimensionOutput_x: int
6. Data
   1. none

### Andere schichten werden in laufe der arbeit noch hinzugefügt

## offene Punkte/Fragen

- Parameter für optimizer in den Layers?
- packen
- und
- entpacken
- predecessor zusätzlich von SchichtNr
- wenn mehrere predecessor wie sind diese angeordnet (wenn pytorch des ned mocht, scheiß ma drauf)
  - Weights werden einfach elementweiße addiert
- trainable weights (Bitmaske für frozen_weights in pytorch) + Layerweise

## Next Steps

1. Klassendiagramm
2. Sequenzdiagramm für Retrainigs Durchlauf
3. Erste Implementierung
4. Nächstes Klassendiagramm

## V1 known and accepted Limitations

- 2DConvolutions
  - Kernel_size, stride, padding and dilation only supported with x = y, The Dataformat supports Parameters with different Hight to Weight Ratio, but he implemantation not
