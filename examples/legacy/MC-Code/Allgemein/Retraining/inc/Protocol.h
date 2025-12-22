
#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

enum class Protocol : uint32_t
{
	None = 0,
	Train = 1,
	Update = 2,
	Safe_and_Swap = 3,
	End_Update = 4,
	New_Sender = 5,
	Write_Flash = 6,
	Finished_Write_Flash = 7,
	Start_Training = 8,
	Inference = 9
};

// pack this struct to guarantee no padding
#pragma pack(push, 1)
typedef struct 
{
	uint32_t LayerNumber;
	uint32_t Weigthamount;
	uint32_t Biasamount;
}Layerinformation_t;
#pragma pack(pop)

#endif