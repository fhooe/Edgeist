#include "Socket_setup.h"
#include "socket.h"
#include "string.h"

#define Port 1000

// network info
static uint8_t gateway[4] = {192, 168, 0, 252};         	// Gateway (e.g., router IP)
static uint8_t subnet[4] = {255, 255, 255, 0};        		// Subnet mask (common for home networks)

// MAC address (unique for your network)
static uint8_t mac[6] = {0x00, 0x08, 0xDC, 0x12, 0x34, 0x00};

// Storage Variables
static ip_t ip_adresses[MAX_BOARDS];
static uint8_t ip_len = 0;
static uint8_t device_pos = 255;
static uint8_t isInit = 0;

static uint8_t GetSn(ip_t ip)
{
	uint8_t pos = 255;
	for(uint8_t i = 0; i < MAX_BOARDS; i++)
	{
		// check ip as uint32_t
		if (*((uint32_t*)ip) == *((uint32_t*)(ip_adresses[i])))
		{
			pos = i;
			break;
		}
	}
	if (pos > device_pos)
	{
		return pos-1;
	}
	return pos;
}

// Check if it should be Server or Client
// return 0 => Server
// return 1 => Client
// return >1 => Error
static uint8_t ServerOrClient(uint8_t pos)
{
	if (pos >= MAX_BOARDS)
	{
		return -1;
	}
	if (pos == device_pos)
	{
		return -2;
	}
	
	uint8_t lowerbit_mask = 0x01;
	
	if ((device_pos & lowerbit_mask) == 0)
	{
		if ((pos & lowerbit_mask) == 0)
		{
			return 1;
		}
		else
		{
			if (pos < device_pos)
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}
	}
	else
	{
		if ((pos & lowerbit_mask) == 0)
		{
			return 0;
		}
		else
		{
			if (pos < device_pos)
			{
				return 0;
			}
			else
			{
				return 1;
			}
		}
	}
}

uint8_t InitSocket(uint8_t pos, ip_t* ips, uint8_t len)
{
	if (pos >= MAX_BOARDS)
	{
		return -1;
	}
	if (ips == 0)
	{
		return -2;
	}
	if (len > MAX_BOARDS || len <= 1)
	{
		return -3;
	}
	device_pos = pos;
	if (device_pos >= MAX_BOARDS)
	{
		// pos is invalid
		return -1;
	}
	
	// modify mac to be unique
	mac[5] = device_pos;
	
	// set w5500 network data
	wiz_NetInfo netinfo;
	wizchip_getnetinfo(&netinfo);
	for(uint8_t idx = 0; idx < 6; idx++)
	{
		netinfo.mac[idx] = mac[idx];
	}
	for(uint8_t idx = 0; idx < 4; idx++)
	{
		netinfo.ip[idx] = ips[device_pos][idx];
		netinfo.gw[idx] = gateway[idx];
		netinfo.sn[idx] = subnet[idx];
	}
	wizchip_setnetinfo(&netinfo);
	
	// store ips
	for (uint8_t i = 0; i < len; i++)
	{
		memcpy(ip_adresses[i], ips[i], sizeof(ips[i]));
	}
	ip_len = len;
	
	isInit = 1;
	return 0;
}

uint8_t OpenServer()
{
	if (isInit == 0)
	{
		return -1;
	}
	uint8_t retval = 255;
	uint8_t sn = 0;
	
	for (uint8_t pos = 0; pos < ip_len; pos++)
	{
		if (pos == device_pos)
		{
			continue;
		}
		
		retval = ServerOrClient(pos);
		if (retval == 0)
		{
			// Create Server SF_IO_NONBLOCK
			retval = socket(sn, Sn_MR_TCP, Port+pos, (SF_IO_NONBLOCK | SF_TCP_NODELAY));
			if (retval != sn)
			{
				while(1);
			}
			
			retval = listen(sn);
			if (retval != SOCK_OK)
			{
					// Handle error, listening failed
					while(1);
			}
		}
		else if (retval != 1)
		{
			// Error
			return retval;
		}
		sn++;
	}
	
	return 0;
}

uint8_t ConnectServer()
{
	if (isInit == 0)
	{
		return -1;
	}
	
	uint8_t retval = 255;
	uint8_t sn = 0;
	
	for (uint8_t pos = 0; pos < ip_len; pos++)
	{
		if (pos == device_pos)
		{
			continue;
		}
		
		retval = ServerOrClient(pos);
		if (retval == 1)
		{
			// Create Client and connect to Server
			retval = socket(sn, Sn_MR_TCP, Port + pos, (SF_IO_NONBLOCK | SF_TCP_NODELAY));
			if (retval != sn)
			{
				while(1);
			}
			
			// trys to connect to client (blocking)
			retval = connect(sn, ip_adresses[pos], Port + pos);
			// SOCK_BUSY because of Non blocking mode
			if (retval != SOCK_OK)
			{
				while(1);
			}
			
		}
		else if (retval != 0)
		{
			// Error
			return retval;
		}
		sn++;
	}
	
	return 0;
}

uint8_t WaitForConnections()
{
	if (isInit == 0)
	{
		return -1;
	}
	
	uint8_t retval = 255;
	uint8_t sn = 0;
	
	for (uint8_t pos = 0; pos < ip_len; pos++)
	{
		if (pos == device_pos)
		{
			continue;
		}
		
		retval = ServerOrClient(pos);
		if (retval == 0)
		{
			uint8_t n_connected = 1;
			while(n_connected)
			{
				if (getSn_SR(sn) == SOCK_ESTABLISHED)
				{
					n_connected = 0;
				}
			}
		}
		sn++;
	}
	return 0;
}

uint32_t send_to(ip_t ip, uint8_t* data, uint32_t data_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	uint8_t sn = GetSn(ip);
	if (sn >= MAX_BOARDS)
	{
		return -2;
	}
	
	return send(sn,data,data_len);
}

uint32_t boradcast(uint8_t* data, uint32_t data_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	uint32_t send_bytes = 0;
	
	for (uint8_t idx = 0; idx < ip_len; idx++)
	{
		if (idx == device_pos)
		{
			continue;
		}
		
		uint8_t sn = GetSn(ip_adresses[idx]);
		
		if (sn >= MAX_BOARDS)
		{
			return -2;
		}
	
	 send_bytes += send(sn,data,data_len);
		
	}
	return send_bytes;
}

uint32_t recv_from(ip_t ip, uint8_t* buffer, uint32_t buffer_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	uint8_t sn = GetSn(ip);
	if (sn >= MAX_BOARDS)
	{
		return -2;
	}
	
	return recv(sn, buffer, buffer_len);
}