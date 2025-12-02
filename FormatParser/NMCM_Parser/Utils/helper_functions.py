import torch
import numpy as np
import struct

class Datatype():
    # static Member
    dtype_map = {
            torch.float32: 0,
            torch.float16: 1,
            torch.float64: 2,
            torch.int32: 3,
            torch.int64: 4,
            # ... and so on
    }

    def __init__(self, type):
        self.type = self.dtype_map.get(type, None)

        if self.type is None:
            raise ValueError("Unknown Type")
    
    def get_number(self) -> int:
        return self.type

    @staticmethod   
    def get_str(val) -> str:
        reverse_lookup = {v: k for k, v in Datatype.dtype_map.items()}
        return str(reverse_lookup.get(val)).split(".")[-1].strip() + "_t"
        
    @staticmethod
    def get_all_datatypes():
        return list(Datatype.dtype_map.values())
        

import struct
import numpy as np

def HexConverter(value, type: str) -> bytes:
    def get_format(t: str):
        match t:
            case "uint8_t": return 'B'
            case "uint16_t": return 'H'
            case "uint32_t": return 'I'
            case "float32_t": return 'f'
            case "bit": return 'bit'
            case _: raise ValueError(f"Error Datatype: {t} unknown")

    fmt = get_format(type)

    if isinstance(value, str):
        value = [ord(c) for c in value]

    # Einzelwert behandeln
    if isinstance(value, (int, float)):
        if fmt == 'f':
            return struct.pack('f', np.float32(value))
        return struct.pack(fmt, value)

    # Bit-Array behandeln
    if fmt == 'bit':
        if len(value) % 8 != 0:
            value += [0] * (8 - len(value) % 8)

        data = bytearray()
        for i in range(0, len(value), 8):
            byte = value[i:i+8]
            byte_value = 0
            for bit in byte:
                byte_value = (byte_value << 1) | bit
            data.append(byte_value)
        return bytes(data)

    # Normale Liste von Werten behandeln
    pack_func = struct.Struct(fmt).pack
    if fmt == 'f':
        packed = b''.join(pack_func(np.float32(val)) for val in value)
    else:
        packed = b''.join(pack_func(val) for val in value)

    return packed




def Sizeof(datatype : str) -> int:
    """
    Returns the size of a datatype in byte
    
    Parameters:
    datatype (str): Name of the datatype

    Returns:
    int: The size of the datatype in byte
    
    Raises:
    Unknown Type: XX: If the datatype is not defined in match case
    """
    match datatype:
        case "uint32_t":
            return 4
        case "uint16_t":
            return 2
        case "uint8_t":
            return 1
        case "float32_t":
            return 4
        case "float16_t":
            return 2
        case _:
            raise "Unknown Type: " + datatype
        
def update_header_offset_length(model_struct, config):
    config["Header"]["Layer_Offset_Table"] = (model_struct[0].data["LayerNrs"],config["Header"]["Layer_Offset_Table"][1])

    return config