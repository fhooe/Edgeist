import importlib.util
import inspect
import os
import pathlib


class HeaderWriter:
    def __init__(self, typefile: str, structfile: str, enumfile: str):
        """
        Constructor of the class H_writter

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

    def writeH(self, config, model_struct):
        """
        Creates or clears the outputfile.
        writes all types and the version from the config
        to the outputfile in c++ style

        Parameters:
        config (dic): data of the configfile as dictionary
        """
        self.__write_Typefile(config)
        self.__write_Structfile(config)
        self.__write_Enumfile(model_struct)

    # Helper function to map the typenames to cpp compatible ones
    def __map_cpp_type(self, typename: str) -> str:
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
            print(f"Warnung: unbekannter Typ '{typename}', wird roh verwendet.")
        return mapping.get(typename, typename)

    def __write_Typefile(self, config):
        with open(self.type_file, "w") as outfile:
            outfile.write("#ifndef " + self.type_filename.upper() + "_H\n")
            outfile.write("#define " + self.type_filename.upper() + "_H\n")
            outfile.write("\n")
            outfile.write("#include<stdint.h>\n")

            for key, _ in config.items():
                if key == "Offset_Table":
                    continue

                if key == "Config-Info":
                    # this section has no types but the version string
                    outfile.write("\n")
                    outfile.write("// " + key + "-Types\n")
                    outfile.write('#define version_str "' + config[key]["Version"][1] + '"\n')
                    outfile.write("typedef " + config[key]["ID"][1] + " ID_t;\n")
                    outfile.write("typedef " + config[key]["Offset_Table"][1] + " Offset_Table_entry;\n")
                    outfile.write("\n")
                else:
                    outfile.write("// " + key + "-Types\n")
                    # write datatypes
                    for name, typeinfo in config[key].items():
                        mapped_type = self.__map_cpp_type(typeinfo[1])
                        outfile.write("\ttypedef " + mapped_type + " " + key + "_" + name + "_t;\n")

                    outfile.write("\n")

            outfile.write("#endif")

    def __write_Structfile(self, config):
        with open(self.struct_file, "w") as outfile:
            outfile.write("#ifndef " + self.struct_filename.upper() + "_H\n")
            outfile.write("#define " + self.struct_filename.upper() + "_H\n")
            outfile.write("\n")
            outfile.write('#include "' + self.type_filename + '.h"\n')
            outfile.write("\n")

            for key, value in config.items():
                if key != "Config-Info" and key != "Offset_Table":
                    outfile.write("// " + key + "-Struct\n")
                    outfile.write("#pragma pack(push, 1)\n")
                    outfile.write("typedef struct{\n")

                    for name, typeinfo in config[key].items():
                        if typeinfo[0] == 1:
                            outfile.write("\t" + key + "_" + name + "_t " + name.lower() + ";\n")
                        elif typeinfo[0] == 0:
                            # should be a pointer
                            outfile.write("\t" + key + "_" + name + "_t* " + name.lower() + ";\n")
                        else:
                            outfile.write(
                                "\t" + key + "_" + name + "_t " + name.lower() + "[" + str(typeinfo[0]) + "];\n"
                            )

                    outfile.write("} Neural_Network_" + key + "_t;\n")
                    outfile.write("#pragma pack(pop)\n")
                    outfile.write("\n")

            outfile.write("#endif")

    # write all IDs of the Layers in ./Layers
    def __write_Enumfile(self, model_struct):
        from nmcm_common.layers.layer import Layer
        from nmcm_common.utils.datatype import Datatype

        types = Datatype.get_all_datatypes()

        with open(self.enum_file, "w") as outfile:
            outfile.write("#ifndef " + self.enum_filename.upper() + "_H\n")
            outfile.write("#define " + self.enum_filename.upper() + "_H\n\n")

            # DataEncodingIDs
            outfile.write("enum class DataEncodingIDs\n{\n")
            for val in types:
                outfile.write("\t " + Datatype.get_str(val).removesuffix("_t") + "_ID = " + str(val) + ",\n")
            outfile.write("};\n\n")

            # LayerIDs – dynamisch aus ./Layers/
            outfile.write("enum class LayerIDs\n{\n")

            layer_dir = pathlib.Path("./Layers")
            for file in layer_dir.glob("*.py"):
                if file.name == "__init__.py":
                    continue

                module_name = file.stem
                spec = importlib.util.spec_from_file_location(module_name, str(file))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, Layer) and obj is not Layer:
                        try:
                            layer_instance = obj(config={})  # evtl. Dummy config anpassen
                            layer_id = getattr(layer_instance, "LayerId", None)
                            if layer_id is not None:
                                outfile.write(f"\t {name}_ID = {layer_id},\n")
                        except Exception as e:
                            print(f"WARNING: Could not initialize class '{name}': '{e}'")

            outfile.write("};\n\n")
            outfile.write("#endif")