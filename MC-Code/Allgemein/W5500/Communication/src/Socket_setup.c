#include "Socket_setup.h"
#include "socket.h"
#include "string.h"
#include "main.h"

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
		if ((pos & lowerbit_mask) != 0)
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

static int8_t Check_for_loopback(ip_t IP_to_check)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	for (uint8_t i = 0; i < 4; i++)
	{
		if (IP_to_check[i] != ip_adresses[device_pos][i])
		{
			return 0;
		}
	}
	return 1;
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
			uint32_t status = 0;
			uint8_t Connection_Retry = 1;
			
			do
			{
				// Create Client and connect to Server
				retval = socket(sn, Sn_MR_TCP, Port + pos, (SF_IO_NONBLOCK | SF_TCP_NODELAY));
				if (retval != sn)
				{
					while(1);
				}
			
				// trys to connect to client 
				retval = connect(sn, ip_adresses[pos], Port + device_pos);
				
				// SOCK_BUSY because of Non blocking mode
				if (retval != SOCK_BUSY)
				{
					while(1);
				}
				
				do
				{
					status = getSn_SR(sn);			
				} while(status == SOCK_INIT || status == SOCK_SYNSENT);
				
				if (status == SOCK_ESTABLISHED)
				{
					// socket has connection
					Connection_Retry = 0;
				}
				else
				{
					// close socket for next try
					close(sn);
					
					// Delay to wait for next Try
					for (int i = 0; i < 1000000; i++)
					{
						__NOP();
					}
				}
			} while (Connection_Retry);
			
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

int32_t send_to(ip_t ip, uint8_t* data, uint32_t data_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	if (data == 0)
	{
		return -2;
	}
	
	if (send_to_direct(ip, (uint8_t*)(&data_len),sizeof(data_len)) != sizeof(data_len))
	{
		return -3;
	}
	
	return send_to_direct(ip,data,data_len);
}

int32_t broadcast(uint8_t* data, uint32_t data_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	if (data == 0)
	{
		return -2;
	}
	
	if (broadcast_direct((uint8_t*)(&data_len),sizeof(data_len)) != sizeof(data_len) * ip_len)
	{
		return -3;
	}
	
	return broadcast_direct(data, data_len);
}

int32_t listen_all(uint8_t* data, uint32_t data_len)
{
	int32_t retval = 0;
	
	if (isInit == 0)
	{
		return -1;
	}

	for (uint8_t i = 0; i < ip_len; i++)
	{
		if (i == device_pos)
		{
			continue;
		}

		if (recv_from_direct(ip_adresses[i], data, data_len) != SOCK_BUSY)
		{
			return i;
		}
	}
	return -1;
}

int32_t recv_from(ip_t ip, uint8_t* buffer, uint32_t buffer_len)
{
	uint32_t length_buffer;
	int32_t retval;
	
	// listen for size
	retval = recv_from_direct(ip, (uint8_t*)(&length_buffer), sizeof(length_buffer));
	
	// check for size
	if (retval != sizeof(length_buffer))	
	{
		return retval;
	}
	
	// wait for header-data
	while (1)
	{
		retval = recv_from_direct(ip, buffer, buffer_len);
		
		if (retval == length_buffer)
		{
			return retval;
		}
		
		if (retval != SOCK_BUSY)
		{
			return retval;
		}
	}
	
}

int32_t send_to_direct_pos(uint8_t pos, uint8_t* data, uint32_t data_len)
{
	if (pos >= ip_len)
	{
		return -1;
	}
	return send_to_direct(ip_adresses[pos], data, data_len);
}

int32_t send_to_direct(ip_t ip, uint8_t* data, uint32_t data_len)
{
	if (isInit == 0)
	{
		return -1;
	}

	// can't send to itself
	if (Check_for_loopback(ip) != 0)
	{
		return 0;
	}
	
	uint8_t sn = GetSn(ip);
	if (sn >= MAX_BOARDS)
	{
		return -2;
	}
	
	if (data == 0)
	{
		return -3;
	}
	
	uint32_t const buffer_size = getSn_TXBUF_SIZE(sn) * 1024;
	int32_t retval = 0;
	uint32_t bytes_send = 0;
	uint8_t tmp=0;
	uint8_t sending = 1;

	while (sending)
	{
		if (data_len > buffer_size)
		{
			retval = send(sn,data,buffer_size);

			if (retval < 0)
			{
				return retval;
			}

			data += retval;
			data_len -= retval;
			bytes_send += retval;
		}
		else
		{
			retval = send(sn,data,data_len);

			if (retval < 0)
			{
				return retval;
			}

			bytes_send += retval;
			sending = 0;
		}

		// implement handshake
		while (1)
		{
			retval = recv(sn,(uint8_t*)bytes_send,sizeof(uint32_t));

			if (retval < 0)
			{
				return retval;
			}

			if (retval == sizeof(uint32_t))
			{
				break;
			}
		}
	}

	return bytes_send;
}

int32_t broadcast_direct(uint8_t* data, uint32_t data_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	if (data == 0)
	{
		return -2;
	}
	
	uint32_t send_bytes = 0;
	
	for (uint8_t idx = 0; idx < ip_len; idx++)
	{
		if (idx == device_pos)
		{
			continue;
		}
	
		send_bytes += send_to_direct(ip_adresses[idx],data,data_len);
		
	}
	return send_bytes;
}
int32_t recv_from_direct_pos(uint8_t pos, uint8_t* buffer, uint32_t buffer_len)
{
	if (pos >= ip_len)
	{
		return -1;
	}
	
	return recv_from_direct(ip_adresses[pos], buffer, buffer_len);
}

int32_t recv_from_direct(ip_t ip, uint8_t* buffer, uint32_t buffer_len)
{
	if (isInit == 0)
	{
		return -1;
	}
	
	// can't listen to itself
	if (Check_for_loopback(ip) != 0)
	{
		return 0;
	}

	uint8_t sn = GetSn(ip);
	if (sn >= MAX_BOARDS)
	{
		return -2;
	}
	
	if (buffer == 0)
	{
		return -3;
	}
	
	int32_t retval = 0;
	retval = recv(sn, buffer, buffer_len);

	if (retval <= 0)
	{
		return retval;
	}

	// handshake
	send(sn, (uint8_t*)&buffer_len, sizeof(uint32_t));

	int32_t datalen = retval;
	
	while (datalen < buffer_len)
	{		
		retval = recv(sn, buffer + datalen, buffer_len - datalen);

		if (retval < 0)
		{
			return retval;
		}
		
		if (retval == 0)
		{
			continue;
		}

		// handshake
		send(sn, (uint8_t*)&retval, sizeof(int32_t));

		datalen += retval;	
	}
	
	
	return datalen;
}