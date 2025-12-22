; Includes Modelinfo File in Flash
; https://developer.arm.com/documentation/ka002916/latest/
	AREA    Modeldata_File_Section, DATA, READONLY
	EXPORT Modeldata
		
; Includes the binary file Model.bin from the current source folder
Modeldata
	INCBIN  ..\..\..\Model\data.bin
Modeldata_End

	EXPORT  Modeldata_End
		
	END