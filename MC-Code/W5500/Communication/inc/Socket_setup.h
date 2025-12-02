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

// Sends Data to IP-Adress
uint32_t send_to(ip_t ip, uint8_t* data, uint32_t data_len);

// Sends Data to all IP-Adresses
uint32_t boradcast(uint8_t* data, uint32_t data_len);

// Looks for Data from IP-Adress
// ip = IP-Adress of sender
// buffer = data destination
// buffer_len = length of the buffer
// return bytes read
uint32_t recv_from(ip_t ip, uint8_t* buffer, uint32_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif