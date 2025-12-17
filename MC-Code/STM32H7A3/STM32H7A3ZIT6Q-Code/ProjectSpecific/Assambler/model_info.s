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
	DCD 	0x00000037
	
SectorNumbers
	DCD		0x00000000
	DCD		0x00000001
	DCD		0x00000002
	DCD		0x00000003
	DCD		0x00000004
	DCD		0x00000005
	DCD		0x00000006
	DCD		0x00000007
	DCD		0x00000008
	DCD		0x00000009
	DCD		0x0000000A
	DCD		0x0000000B
	DCD		0x0000000C
	DCD		0x0000000D
	DCD		0x0000000E
	DCD		0x0000000F
	DCD		0x00000010
	DCD		0x00000011
	DCD		0x00000012
	DCD		0x00000013
	DCD		0x00000014
	DCD		0x00000015
	DCD		0x00000016
	DCD		0x00000017
	DCD		0x00000018
	DCD		0x00000019
	DCD		0x0000001A
	DCD		0x0000001B
	DCD		0x0000001C
	DCD		0x0000001D
	DCD		0x0000001E
	DCD		0x0000001F
	DCD		0x00000020
	DCD		0x00000021
	DCD		0x00000022
	DCD		0x00000023
	DCD		0x00000024
	DCD		0x00000025
	DCD		0x00000026
	DCD		0x00000027
	DCD		0x00000028
	DCD		0x00000029
	DCD		0x0000002A
	DCD		0x0000002B
	DCD		0x0000002C
	DCD		0x0000002D
	DCD		0x0000002E
	DCD		0x0000002F
	DCD		0x00000030
	DCD		0x00000031
	DCD		0x00000032
	DCD		0x00000033
	DCD		0x00000034
	DCD		0x00000035
	DCD		0x00000036
	
	EXPORT SectorAmount
	EXPORT SectorNumbers
		
	END