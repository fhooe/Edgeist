
import json

class JSON_writer:
    def __init__(self, filename : str):
        """
        Constructor of the class JSON_writter
        
        Checks fileending is valid .json
        
        Parameters:
        filename (str): filename and path e.g. ./test.json
        
        Raises:
        File is no .json: If file ends not with .json
        """

        # init json File
        self.filename = filename

        if self.filename.split(".")[-1] != "json":
            raise("File is no .json")

    def writeJSON(self, model_struct):
        """
        Creates or clears the outputfile.
        writes all model information to the outputfile
        in json format.
        
        Parameters:
        model_struct (array): list of header + all layers
        """
        idx = 1
        # create or clear the file
        with open(self.filename, "w") as outfile:
            # write start symbol
            outfile.write("[\n")
            for layer in model_struct:
                # Serializing json
                json_object = json.dumps(layer.json_data, indent=4)
            
                # Writing to output file
                outfile.write(json_object)
                if idx == len(model_struct):
                    outfile.write("\n]")
                else:
                    outfile.write(",\n")

                idx += 1
        