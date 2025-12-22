
#ifndef FLASH_MANAGER_H
#define FLASH_MANAGER_H

#include "stm32f4xx_hal.h"

enum class Flash_Returntypes
{
	Flash_OK = 0,
	Flash_send_ack = 1,
	
	Flash_Error = -1,
	Flash_wrong_addr = -2,
	Flash_Nullptr = -3,
};

class Flash_manager
{
	public:
		// functions
		Flash_manager();
	
		Flash_Returntypes WriteFlash(uint32_t addr, uint32_t* data, int32_t data_size);
	
		Flash_Returntypes ReadFlash(uint32_t addr, uint8_t* buffer, uint32_t bytes);
	
		Flash_Returntypes EraseFlash(uint32_t start_addr, uint32_t Sector_amount, uint32_t* Sector_Nr);

		Flash_Returntypes Flush();
};

#endif