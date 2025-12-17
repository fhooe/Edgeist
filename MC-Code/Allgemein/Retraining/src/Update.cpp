#include "Update.h"
#include "flash_manager.h"
#include "Socket_setup.h"
#include "Protocol.h"

void Reset_IsReady(uint8_t pos);
bool Check_IsReady(uint8_t amount);

uint8_t IsReady[MAX_BOARDS];

#define Header_length 7
#define Data_length 1024

int32_t Update_Model(uint8_t pos, uint8_t amount)
{
	if (amount == 0 || pos >= amount)
	{
		return -1;
	}
	
	// Variables
	bool sender = (pos == 0);
	bool update = 1;
	uint32_t* pTmp = 0;
	uint32_t data_len = 0;
	uint8_t* pLayer_pos = 0;
	uint32_t Layer_pos32 = 0;
	int32_t socket_pos = -1;
	
	Flash_manager fm = Flash_manager();
	
	uint8_t Header_buffer[Header_length];
	uint8_t Data_buffer[Data_length];
	
	Reset_IsReady(pos);
	
	while(update)
	{
		if (sender)
		{
			// TODO: get values
			uint32_t layer_amount = 3;
			uint32_t layer_ids[3] = {1,2,3};
			
			// for each layer on this device
			for (uint8_t i = 0; i < layer_amount; i++)
			{
				// TODO: Length of the Data in the current layer
				data_len = 2048;
				// TODO: set pointer to the start of the layer data
				pLayer_pos = Data_buffer;
				
				// Init Header
				Header_buffer[0] = static_cast<uint8_t>(Protocol::Write_Flash);
				pTmp = reinterpret_cast<uint32_t*>(Header_buffer + 1);
				*pTmp = data_len;
				Header_buffer[5] = data_len/Data_length;
				if (data_len % Data_length != 0)
				{
					Header_buffer[5]++;
				}
				Header_buffer[6] = layer_ids[i];
				// send Header mit layer information
				broadcast(Header_buffer,Header_length);
				
				
				for (uint32_t i = 0; i < data_len; i += Data_length)
				{
					if (i + Data_length < data_len)
					{
						broadcast(pLayer_pos,Data_length);
						pLayer_pos += Data_length;
					}
					else
					{
						broadcast(pLayer_pos,data_len-i);
					}
					
					// wait for all other devices to write data into flash
					while (!Check_IsReady(amount))
					{
						socket_pos = listen_all(Data_buffer, 1);
						if (socket_pos >= 0)
						{
							IsReady[socket_pos] = 1;
						}
					}
					Reset_IsReady(pos);
				}
			}
			// send info to next or terminate update process
			if (pos + 1 == amount)
			{
				Header_buffer[0] = static_cast<uint8_t>(Protocol::End_Update);
				broadcast_direct(Header_buffer,Header_length);
				update = 0;
			}
			else
			{
				Header_buffer[0] = static_cast<uint8_t>(Protocol::New_Sender);
				send_to_direct_pos(pos+1,Header_buffer,Header_length);
				sender = false;
			}
		}
		else
		{
			socket_pos = listen_all(Header_buffer, Header_length);
			
			// vaild message
			if (socket_pos >= 0)
			{
				switch (Header_buffer[0])
				{
					case static_cast<uint8_t>(Protocol::End_Update):
						// end update
						update = 0;
						break;
					case static_cast<uint8_t>(Protocol::New_Sender):
						// new sender
						sender = true;
						break;
					case static_cast<uint8_t>(Protocol::Write_Flash):
						uint32_t recv_len = 0;
					
						// TODO: set Layerpos (LayerID = Header_buffer[6])
						Layer_pos32 = 0x800000;
					
						for (uint8_t i = 0; i < Header_buffer[5]; i++)
						{
							if (i + 1 == Header_buffer[5])
							{
								// last data buffer
								recv_len = (*reinterpret_cast<uint32_t*>(Header_buffer+1)) - Data_length*i; 
								recv_from_direct_pos(socket_pos, Data_buffer, recv_len);
							}
							else
							{
								// full data buffer
								recv_len = Data_length;
								recv_from_direct_pos(socket_pos, Data_buffer, recv_len);
							}
							
							// write Data into flash
							fm.WriteFlash(Layer_pos32, reinterpret_cast<uint32_t*>(Data_buffer), recv_len/4);
							Layer_pos32 += recv_len;
							
							// send Write finished to sender
							Header_buffer[0] = static_cast<uint8_t>(Protocol::Finished_Write_Flash);
							send_to_direct_pos(socket_pos, Header_buffer, Header_length);
						}
						
						break;
				}
			}
		}
	}
	
	return 0;
}

void Reset_IsReady(uint8_t pos)
{
	for (uint8_t i = 0; i < MAX_BOARDS; i++)
	{
		IsReady[i] = (pos == i) ? 1 : 0;
	}
}

bool Check_IsReady(uint8_t amount)
{
	for (uint8_t i = 0; i < amount; i++)
	{
		if (IsReady[i] == 0)
		{
			return false;
		}
	}
	return true;
}