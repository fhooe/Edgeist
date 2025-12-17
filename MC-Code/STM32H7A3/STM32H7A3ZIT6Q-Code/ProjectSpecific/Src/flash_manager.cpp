
#include "flash_manager.h"
#include <string.h>

Flash_manager::Flash_manager()
{

}

Flash_Returntypes Flash_manager::EraseFlash(uint32_t start_addr, uint32_t Sector_amount, uint32_t* Sector_Nr)
{
	if (start_addr == 0 || Sector_Nr == 0 || Sector_amount == 0)
	{
		return Flash_Returntypes::Flash_Nullptr;
	}
	
	// check if addr is in the flash
	if (!IS_FLASH_PROGRAM_ADDRESS(start_addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	
	// Init Erase Struct
	FLASH_EraseInitTypeDef erase_struct;
	erase_struct.TypeErase = FLASH_TYPEERASE_SECTORS;
	
	erase_struct.NbSectors = 1;
	
#if defined (DUAL_BANK)
	if (IS_FLASH_PROGRAM_ADDRESS_BANK1(start_addr))
	{
		erase_struct.Banks = FLASH_BANK_1;
	}
	else if (IS_FLASH_PROGRAM_ADDRESS_BANK2(start_addr))
	{
		erase_struct.Banks = FLASH_BANK_2;
	}
	else
	{
		return Flash_Returntypes::Flash_Error;
	}
#else
	if (IS_FLASH_PROGRAM_ADDRESS_BANK1(start_addr))
	{
		erase_struct.Banks = FLASH_BANK_1;
	}
	else
	{
		return Flash_Returntypes::Flash_Error;
	}
#endif
	
	
#if defined(FLASH_CR_PSIZE)
	erase_struct.VoltageRange = FLASH_VOLTAGE_RANGE_3;
#endif

	uint32_t SectorError;
	uint32_t* local_Sector_Nr = Sector_Nr;
	for (uint32_t i = 0; i < Sector_amount; i++)
	{
		erase_struct.Sector = *local_Sector_Nr;
		
		if (HAL_FLASHEx_Erase(&erase_struct, &SectorError) != HAL_OK)
		{
			return Flash_Returntypes::Flash_Error;
		}
		
		if (SectorError != 0xFFFFFFFF)
		{
			return Flash_Returntypes::Flash_Error;
		}
	}
	
	return Flash_Returntypes::Flash_OK;
}

Flash_Returntypes Flash_manager::WriteFlash(uint32_t addr, uint32_t* data, uint32_t data_size)
{
	if (addr == 0 || data == 0)
	{
		return Flash_Returntypes::Flash_Nullptr;
	}
	
	if (!IS_FLASH_PROGRAM_ADDRESS(addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	uint32_t Flash_reset = 0xFFFFFFFF;
	
	// need to look for a define (128 bit write length)
	// different for every mc
	uint8_t const Flash_Write_Length = 4;
	
	// define data (needs to be 32 bit alignet) and can only write (128 bit)
	uint32_t Transmit_data[Flash_Write_Length] __attribute__((aligned(4))) = {Flash_reset};
	
	// current pos in transmit data
	uint32_t pos = 0;
	
	
	// First Transmittion if addr is not alignt
	uint8_t offset = addr % (Flash_Write_Length * sizeof(uint32_t));
	
	if (offset != 0)
	{
		// new data to write
		for (uint8_t i = offset; i < Flash_Write_Length; i++)
		{
			Transmit_data[i] = data[pos];
			pos++;
		}
		
		// old data from flash
		for (int8_t i = Flash_Write_Length - offset - 1; i < 0; i--)
		{
			addr -= sizeof(uint32_t);
			this->ReadFlash(addr, reinterpret_cast<uint8_t*>(Transmit_data + i), sizeof(uint32_t));
		}
		
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_FLASHWORD, reinterpret_cast<uint32_t>(addr), reinterpret_cast<uint32_t>(Transmit_data)) != HAL_OK)
		{
			return Flash_Returntypes::Flash_Error;
		}
		
		// now addr is Flash_Write_Length aligent
		addr += (Flash_Write_Length * sizeof(uint32_t));
	}
		
	
	for (; pos < data_size - Flash_Write_Length - 1; pos+=Flash_Write_Length)
	{
		for (uint8_t idx = 0; idx < Flash_Write_Length; idx++)
		{
			Transmit_data[idx] = data[pos+idx];
		}
		
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_FLASHWORD, reinterpret_cast<uint32_t>(addr), reinterpret_cast<uint32_t>(Transmit_data)) != HAL_OK)
		{
			return Flash_Returntypes::Flash_Error;
		}
		
		addr += (Flash_Write_Length * sizeof(uint32_t));
	}
	
	// transmit last data
	if (pos < data_size)
	{
		uint8_t idx = 0;
		for (; pos < data_size; pos++)
		{
			Transmit_data[idx] = data[pos];
			idx++;
			addr += sizeof(uint32_t);
		}
		for (; idx < Flash_Write_Length; idx++)
		{
			this->ReadFlash(addr,	reinterpret_cast<uint8_t*>(Transmit_data + idx), sizeof(uint32_t));
			addr += sizeof(uint32_t);
		}
		
		// decrement addr* because data gen incrementet it
		addr -= (Flash_Write_Length * sizeof(uint32_t));
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_FLASHWORD, reinterpret_cast<uint32_t>(addr), reinterpret_cast<uint32_t>(Transmit_data)) != HAL_OK)
		{
			return Flash_Returntypes::Flash_Error;
		}
	}
	
	return Flash_Returntypes::Flash_OK;
}

Flash_Returntypes Flash_manager::ReadFlash(uint32_t addr, uint8_t* buffer, uint32_t bytes)
{
	if (addr == 0 || buffer == 0)
	{
		return Flash_Returntypes::Flash_Nullptr;
	}
	
	// check if addr is in the flash
	if (!IS_FLASH_PROGRAM_ADDRESS(addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	
	const uint8_t* flash_ptr = (const uint8_t*)addr;
	memcpy(buffer, flash_ptr, bytes);
	
	return Flash_Returntypes::Flash_OK;
}