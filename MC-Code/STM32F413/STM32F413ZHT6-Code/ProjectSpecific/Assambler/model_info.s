; Includes Modelinfo File in Flash
; https://developer.arm.com/documentation/ka002916/latest/
	AREA    Modelinfo_File_Section, DATA, READONLY

	EXPORT  Modelinfo

; Includes the binary file Model.bin from the current source folder
Modelinfo
	INCBIN  ..\..\..\Model\model.bin
Modelinfo_End

; define a constant which contains the size of the image above
Modelinfo_length
	DCD     Modelinfo_End - Modelinfo

	EXPORT  Modelinfo_length
		
; define the flash sectors used in scatterfile for trainable data
SectorAmount
	DCD 	0x00000004
	
SectorNumbers
	DCD		0x0000000C
	DCD		0x0000000D
	DCD		0x0000000E
	DCD		0x0000000F
	
	EXPORT SectorAmount
	EXPORT SectorNumbers
		
	END