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
	DCD 	0x00000008
	
SectorNumbers
	DCD		0x00000000
	DCD		0x00000001
	DCD		0x00000002
	DCD		0x00000003
	DCD		0x00000004
	DCD		0x00000005
	DCD		0x00000006
	DCD		0x00000007
	
	EXPORT SectorAmount
	EXPORT SectorNumbers
		
	END