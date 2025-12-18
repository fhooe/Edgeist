def Sizeof(datatype: str) -> int:
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
    config["Header"]["Layer_Offset_Table"] = (
        model_struct[0].data["LayerNrs"],
        config["Header"]["Layer_Offset_Table"][1],
    )

    return config
