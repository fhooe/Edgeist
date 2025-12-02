
import struct

def BinaryConverter(binary, datatype : str):

    match datatype:
        case "uint8_t":
            data = struct.unpack('B', binary)[0]
        case "uint16_t":
            data = struct.unpack('H', binary)[0]
        case "uint32_t":
            data = struct.unpack('I', binary)[0]
        case "float32_t":
            data = struct.unpack('f', binary)[0]
        case _:
            raise "Error Datatype: " + type + " unknown"
        
    return data