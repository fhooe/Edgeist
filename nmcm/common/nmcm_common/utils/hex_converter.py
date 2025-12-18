import struct

import numpy as np


def HexConverter(value, type: str) -> bytes:
    def get_format(t: str):
        match t:
            case "uint8_t":
                return "B"
            case "uint16_t":
                return "H"
            case "uint32_t":
                return "I"
            case "float32_t":
                return "f"
            case "bit":
                return "bit"
            case _:
                raise ValueError(f"Error Datatype: {t} unknown")

    fmt = get_format(type)

    if isinstance(value, str):
        value = [ord(c) for c in value]

    # Einzelwert behandeln
    if isinstance(value, (int, float)):
        if fmt == "f":
            return struct.pack("f", np.float32(value))
        return struct.pack(fmt, value)

    # Bit-Array behandeln
    if fmt == "bit":
        if len(value) % 8 != 0:
            value += [0] * (8 - len(value) % 8)

        data = bytearray()
        for i in range(0, len(value), 8):
            byte = value[i : i + 8]
            byte_value = 0
            for bit in byte:
                byte_value = (byte_value << 1) | bit
            data.append(byte_value)
        return bytes(data)

    # Treat normal list of values
    pack_func = struct.Struct(fmt).pack
    if fmt == "f":
        packed = b"".join(pack_func(np.float32(val)) for val in value)
    else:
        packed = b"".join(pack_func(val) for val in value)

    return packed
