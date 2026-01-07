import importlib.util
import inspect
import os
import pathlib


class HeaderWriter:
    #: defines the spaces used instead of a tab as according to the clang-format rules.
    _TAB = "    "

    def __init__(self, typefile: str, structfile: str, enumfile: str):
        """
        Constructor of the class HeaderWriter

        Checks fileending is valid .h

        Parameters:
        file (str): filename and path e.g. ./test.h

        Raises:
        File is no .h: If file ends not with .h
        """
        # init h File
        self.type_file = typefile
        self.type_filename = os.path.basename(typefile).split(".")[0]

        if self.type_file.split(".")[-1] != "h":
            raise ValueError(f"Type-file '{self.type_file}' is missing the '.h' file extension")

        self.struct_file = structfile
        self.struct_filename = os.path.basename(structfile).split(".")[0]

        if self.struct_file.split(".")[-1] != "h":
            raise ValueError(f"Struct-file '{self.struct_file}' is missing the '.h' file extension")

        self.enum_file = enumfile
        self.enum_filename = os.path.basename(enumfile).split(".")[0]

        if self.enum_file.split(".")[-1] != "h":
            raise ValueError(f"Enum-file '{self.enum_file}' is missing the '.h' file extension")

    def write(self, config, model_struct):
        """
        Creates or clears the outputfile.
        writes all types and the version from the config
        to the outputfile in c++ style

        Parameters:
        config (dic): data of the configfile as dictionary
        """
        self._write_type_header(config)
        self._write_struct_header(config)
        self._write_enum_header(model_struct)

    # Helper function to map the typenames to cpp compatible ones
    def _map_to_cpp_type(self, typename: str) -> str:
        mapping = {
            "float32_t": "float",
            "float64_t": "double",
            "float16_t": "uint16_t",
            "int8_t": "int8_t",
            "uint8_t": "uint8_t",
            "int16_t": "int16_t",
            "uint16_t": "uint16_t",
            "int32_t": "int32_t",
            "uint32_t": "uint32_t",
            "int64_t": "int64_t",
            "uint64_t": "uint64_t",
        }
        if typename not in mapping:
            raise ValueError(f"Unknown type '{typename}'")
        return mapping.get(typename, typename)

    def _write_type_header(self, config):
        with open(self.type_file, "w") as outfile:
            outfile.write(f"#ifndef {self.type_filename.upper()}_H\n")
            outfile.write(f"#define {self.type_filename.upper()}_H\n")
            outfile.write("\n")
            outfile.write("#include <cstdint>\n")

            for key, _ in config.items():
                if key == "Offset_Table":
                    continue

                if key == "Config-Info":
                    # this section has no types but the version string
                    outfile.write("\n")
                    outfile.write(f"// {key}-Types\n")
                    outfile.write(f'static constexpr auto* VERSION_STR = "' + config[key]["Version"][1] + '";\n')
                    outfile.write(f"using ID_t = {config[key]["ID"][1]};\n")
                    outfile.write(f"using Offset_Table_entry = {config[key]["Offset_Table"][1]};\n")
                    outfile.write("\n")
                else:
                    outfile.write(f"// {key}-Types\n")
                    # write datatypes
                    for name, typeinfo in config[key].items():
                        mapped_type = self._map_to_cpp_type(typeinfo[1])
                        outfile.write(f"using {key}_{name}_t = {mapped_type};\n")

                    outfile.write("\n")

            outfile.write(f"#endif // {self.type_filename.upper()}_H")

    def _write_struct_header(self, config):
        with open(self.struct_file, "w") as outfile:
            outfile.write(f"#ifndef {self.struct_filename.upper()}_H\n")
            outfile.write(f"#define {self.struct_filename.upper()}_H\n")
            outfile.write("\n")
            outfile.write(f'#include "{self.type_filename}.h"\n')
            outfile.write("\n")

            for key, _ in config.items():
                if key != "Config-Info" and key != "Offset_Table":
                    outfile.write(f"// {key}-Struct\n")
                    outfile.write("#pragma pack(push, 1)\n")
                    outfile.write("struct Neural_Network_" + key + "_t {\n")

                    for name, typeinfo in config[key].items():
                        if typeinfo[0] == 1:
                            outfile.write(f"{HeaderWriter._TAB}{key}_{name}_t {name.lower()};\n")
                        elif typeinfo[0] == 0:
                            # should be a pointer
                            outfile.write(f"{HeaderWriter._TAB}{key}_{name}_t* {name.lower()};\n")
                        else:
                            outfile.write(f"{HeaderWriter._TAB}{key}_{name}_t {name.lower()} [{str(typeinfo[0])}];\n")

                    outfile.write("};\n")
                    outfile.write("#pragma pack(pop)\n")
                    outfile.write("\n")

            outfile.write("#endif")

    # write all IDs of the Layers in ./Layers
    def _write_enum_header(self, model_struct):
        from nmcm_common.layers import LAYERS
        from nmcm_common.utils.datatype import Datatype

        types = Datatype.get_all_datatypes()

        with open(self.enum_file, "w") as outfile:
            outfile.write(f"#ifndef {self.enum_filename.upper()}_H\n")
            outfile.write(f"#define {self.enum_filename.upper()}_H\n\n")

            # DataEncodingIDs
            outfile.write("enum class DataEncodingIDs {\n")
            for val in types:
                outfile.write(f"{HeaderWriter._TAB}{Datatype.get_str(val).removesuffix("_t")}_ID = {str(val)},\n")
            outfile.write("};\n\n")

            outfile.write("enum class LayerIDs {\n")

            for layer in LAYERS:
                outfile.write(f"{HeaderWriter._TAB}{layer.__name__}_ID = {layer.LAYER_ID},\n")

            outfile.write("};\n\n")
            outfile.write("#endif")
