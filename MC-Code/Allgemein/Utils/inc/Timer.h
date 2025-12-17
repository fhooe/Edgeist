#ifndef TIMER_H
#define TIMER_H

#include "main.h"

// Starts the given timer at counter = start_val
void Start_timer(TIM_HandleTypeDef& timer, uint32_t start_val = 0);

// Reads the Counter and Stops the timer
// returns the Countervalue = Ticks
uint32_t Stop_timer(TIM_HandleTypeDef& timer);

// Only reads the Counter of the timer
// returns the Countervalue = Ticks
uint32_t GetCount_timer(TIM_HandleTypeDef& timer);

#endif