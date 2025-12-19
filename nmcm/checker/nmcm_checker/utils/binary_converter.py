import struct


def BinaryConverter(binary, datatype: str):
    match datatype:
        case "uint8_t":
            return struct.unpack("B", binary)[0]
        case "uint16_t":
            return struct.unpack("H", binary)[0]
        case "uint32_t":
            return struct.unpack("I", binary)[0]
        case "float32_t":
            return struct.unpack("f", binary)[0]
        case _:
            raise ValueError(f"Unknown type '{type}'")
