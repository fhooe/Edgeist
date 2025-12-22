#ifndef _SOCKET_SETUP_H_
#define _SOCKET_SETUP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "w5500.h"
#include <stdint.h>

// the w5500 can only have 8 socket-connections
#define MAX_BOARDS (_WIZCHIP_SOCK_NUM_ + 1)

typedef uint8_t ip_t[4];

uint8_t InitSocket(uint8_t pos, ip_t* ips, uint8_t len);

// Opens all needed Server
// pos = position of the board (need to be smaler than MAX_BOARDS)
// ips = is a array of ip-adresses 
// len = amount of ip-adresses
uint8_t OpenServer();

// Connects to all needed Servers as Client
// pos = position of the board (need to be smaler than MAX_BOARDS)
// ips = is a array of ip-adresses 
// len = amount of ip-adresses
uint8_t ConnectServer();

// Waits for all Clients to connect to the Servers
uint8_t WaitForConnections();

//***********************************************************************
// this Functions send the length first and than the buffer (2 Messages)
//***********************************************************************

// Sends Data to IP-Adress
int32_t send_to(ip_t ip, uint8_t* data, uint32_t data_len);

// Sends Data to all IP-Adresses
int32_t broadcast(uint8_t* data, uint32_t data_len);

int32_t listen_all(uint8_t* data, uint32_t data_len);

// Looks for Data from IP-Adress
// ip = IP-Adress of sender
// buffer = data destination
// buffer_len = length of the buffer
// return bytes read
int32_t recv_from(ip_t ip, uint8_t* buffer, uint32_t buffer_len);

//***********************************************************************
// direct-Functions send the buffer to the IP
//***********************************************************************

// Sends Data to IP-Adress
int32_t send_to_direct_pos(uint8_t pos, uint8_t* data, uint32_t data_len);
int32_t send_to_direct(ip_t ip, uint8_t* data, uint32_t data_len);

// Sends Data to all IP-Adresses
int32_t broadcast_direct(uint8_t* data, uint32_t data_len);

// Looks for Data from IP-Adress
// ip = IP-Adress of sender
// buffer = data destination
// buffer_len = length of the buffer
// return bytes read
int32_t recv_from_direct_pos(uint8_t pos, uint8_t* buffer, uint32_t buffer_len);
int32_t recv_from_direct(ip_t ip, uint8_t* buffer, uint32_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif