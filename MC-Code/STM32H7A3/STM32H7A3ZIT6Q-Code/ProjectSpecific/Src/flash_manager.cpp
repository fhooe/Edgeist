
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

	HAL_FLASH_Unlock();
	
	uint32_t SectorError;
	uint32_t* local_Sector_Nr = Sector_Nr;
	for (uint32_t i = 0; i < Sector_amount; i++)
	{
		erase_struct.Sector = *local_Sector_Nr;
		
		if (HAL_FLASHEx_Erase(&erase_struct, &SectorError) != HAL_OK)
		{
			HAL_FLASH_Lock();
			return Flash_Returntypes::Flash_Error;
		}
		
		if (SectorError != 0xFFFFFFFF)
		{
			HAL_FLASH_Lock();
			return Flash_Returntypes::Flash_Error;
		}
		local_Sector_Nr++;
	}
	
	HAL_FLASH_Lock();
	
	return Flash_Returntypes::Flash_OK;
}

Flash_Returntypes Flash_manager::WriteFlash(uint32_t addr, uint32_t* data, int32_t data_size)
{
	if (addr == 0 || data == 0)
	{
		return Flash_Returntypes::Flash_Nullptr;
	}
	
	if (!IS_FLASH_PROGRAM_ADDRESS(addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	
	if (data_size <= 0)
	{
		return Flash_Returntypes::Flash_Error;
	}
	// current pos in transmit data
	int32_t pos = 0;
	
	// unlock Flash
	HAL_FLASH_Unlock();
	
	// First Transmittion if addr is not alignt
	// >> 2 = / sizeof(uint32_t) 
	int8_t offset = (addr % (Flash_Write_Length * sizeof(uint32_t))) >> 2;
	
	if (offset != 0)
	{
		// new data to write
		for (uint8_t i = offset; i < Flash_Write_Length; i++)
		{
			this->Transmit_data[i] = data[pos];
			pos++;
		}
		
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_FLASHWORD, this->addr, reinterpret_cast<uint32_t>(this->Transmit_data)) != HAL_OK)
		{
			HAL_FLASH_Lock();
			return Flash_Returntypes::Flash_Error;
		}
		// reset Address
		this->addr = 0;
		
		// now addr is Flash_Write_Length aligent
		addr += (Flash_Write_Length * sizeof(uint32_t));
	}
		
	
	for (; pos <= (data_size - Flash_Write_Length); pos+=Flash_Write_Length)
	{
		for (uint8_t idx = 0; idx < Flash_Write_Length; idx++)
		{
			this->Transmit_data[idx] = data[pos+idx];
		}
		
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_FLASHWORD, reinterpret_cast<uint32_t>(addr), reinterpret_cast<uint32_t>(this->Transmit_data)) != HAL_OK)
		{
			HAL_FLASH_Lock();
			return Flash_Returntypes::Flash_Error;
		}
		
		addr += (Flash_Write_Length * sizeof(uint32_t));
	}
	
	// safe last data
	if (pos < data_size)
	{
		this->addr = addr;
		uint8_t idx = 0;
		for (; pos < data_size; pos++)
		{
			this->Transmit_data[idx] = data[pos];
			idx++;
		}
		for (; idx < this->Flash_Write_Length; idx++)
		{
			this->Transmit_data[idx] = 0xFFFFFFFF;
		}
	}
	
	HAL_FLASH_Lock();
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

Flash_Returntypes Flash_manager::Flush()
{
	if (this->addr != 0)
	{
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_FLASHWORD, this->addr, reinterpret_cast<uint32_t>(this->Transmit_data)) != HAL_OK)
		{
			HAL_FLASH_Lock();
			return Flash_Returntypes::Flash_Error;
		}
	}
	return Flash_Returntypes::Flash_OK;
}