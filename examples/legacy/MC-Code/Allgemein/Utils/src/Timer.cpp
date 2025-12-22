# include "Timer.h"

void Start_timer(TIM_HandleTypeDef& timer, uint32_t start_val)
{
    __HAL_TIM_SET_COUNTER(&timer, start_val);
    HAL_TIM_Base_Start(&timer);
}

uint32_t Stop_timer(TIM_HandleTypeDef& timer)
{
    uint32_t ticks = __HAL_TIM_GET_COUNTER(&timer);
		HAL_TIM_Base_Stop(&timer);

    return  ticks;
}

uint32_t GetCount_timer(TIM_HandleTypeDef& timer)
{
    return __HAL_TIM_GET_COUNTER(&timer);
}