
#include "flash_manager.h"
#include <string.h>

Flash_manager::Flash_manager()
{
		__HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP | FLASH_FLAG_WRPERR | FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
}

Flash_Returntypes Flash_manager::EraseFlash(uint32_t start_addr, uint32_t Sector_amount, uint32_t* Sector_Nr)
{
	if (start_addr == 0 || Sector_Nr == 0 || Sector_amount == 0)
	{
		return Flash_Returntypes::Flash_Nullptr;
	}
	
	// check if addr is in the flash
	if (!IS_FLASH_ADDRESS(start_addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	
	__HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP | FLASH_FLAG_WRPERR | FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
	
	// Init Erase Struct
	FLASH_EraseInitTypeDef erase_struct;
	erase_struct.TypeErase = FLASH_TYPEERASE_SECTORS;
	
	erase_struct.NbSectors = 1;
	
	erase_struct.Banks = FLASH_BANK_1;
	
	erase_struct.VoltageRange = FLASH_VOLTAGE_RANGE_1;

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
	
	if (!IS_FLASH_ADDRESS(addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	
	if (data_size <= 0)
	{
		return Flash_Returntypes::Flash_Error;
	}
	
	__HAL_FLASH_CLEAR_FLAG(FLASH_FLAG_EOP | FLASH_FLAG_WRPERR | FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
	
	// unlock Flash
	HAL_FLASH_Unlock();
	
	for (int32_t pos = 0; pos < data_size; pos++)
	{
		if (HAL_FLASH_Program(FLASH_TYPEPROGRAM_WORD, reinterpret_cast<uint32_t>(addr), reinterpret_cast<uint32_t>(*(data + pos))) != HAL_OK)
		{
			HAL_FLASH_Lock();
			return Flash_Returntypes::Flash_Error;
		}
		
		addr += sizeof(uint32_t);
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
	if (!IS_FLASH_ADDRESS(addr))
	{
		return Flash_Returntypes::Flash_wrong_addr;
	}
	
	const uint8_t* flash_ptr = (const uint8_t*)addr;
	memcpy(buffer, flash_ptr, bytes);
	
	return Flash_Returntypes::Flash_OK;
}

Flash_Returntypes Flash_manager::Flush()
{
	// can write 32 doesn't need it
	return Flash_Returntypes::Flash_OK;
}