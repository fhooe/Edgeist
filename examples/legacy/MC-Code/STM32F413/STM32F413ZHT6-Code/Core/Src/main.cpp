/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/

// Board includes
#include "main.h"
#include "spi.h"
#include "gpio.h"
#include "tim.h"

// Communication includes
#include "w5500_spi.h"
#include "w5500.h"
#include "socket.h"
#include "Socket_setup.h"

// Model includes
#include "pch.h"
#include "model.h"
#include "ErrorTypes.h"
#include "OptimizerDataTypes.h"
#include "mnist_loader.h"

#include "Protocol.h"
#include "timer.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define BUFFER_SIZE 8192
	
extern const uint8_t Modelinfo[];
extern const uint32_t Modelinfo_length;

extern const uint32_t SectorAmount;
extern const uint8_t SectorNumbers[];

extern const uint8_t Modeldata[];
extern const uint8_t Modeldata_End[];

extern uint8_t Heap_Mem[];
const uint32_t Heap_Size = 0x3A000;

extern uint8_t Stack_Mem[];
const uint32_t Stack_Size = 0xFF00;
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
static uint8_t buffer[BUFFER_SIZE];

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
	uint8_t* start_data = const_cast<uint8_t*>(Modeldata);
	uint32_t start_data_32 = reinterpret_cast<uint32_t>(reinterpret_cast<uint32_t*>(start_data));
  /* USER CODE BEGIN 1 */

	// init Heap with 0xFF to check usage 
	uint8_t* pHeap = Heap_Mem;
	for (uint32_t i = Heap_Size/4; i < Heap_Size; i++)
	{
		Heap_Mem[i] = 0xFF;
	}
	
	// init Stack with 0xFF to check usage
	uint8_t* pStack = Stack_Mem;
	for (uint32_t i = 0; i < (Stack_Size * 3) /4; i++)
  {
		Stack_Mem[i] = 0xFF;
  }
	
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_SPI1_Init();
	MX_TIM2_Init();
	
	Flash_manager fm = Flash_manager();
	wizchip_init(NULL, NULL); // default buffer size (2kb)
	W5500_Init();
	
  /* USER CODE BEGIN 2 */
	uint8_t pos = 1;
	ip_t IPs[] = {{192,168,0,29},{192,168,0,30}};
	uint8_t num_ip = 2;
	
	com_t CommunicationData;
	CommunicationData.IP_Addresses = IPs;
	CommunicationData.size = num_ip;
	CommunicationData.pos = pos;

	// start socket
	InitSocket(pos, IPs, 2); 
	
	OpenServer();
	
	ConnectServer();
	
	WaitForConnections();

	
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

	// Implement training
	void* voidPtr_fixed = const_cast<void*>(static_cast<const void*>(Modelinfo));
	void* voidPtr_trainable = const_cast<void*>(static_cast<const void*>(Modeldata));
	
	uint32_t Start = 4;
	uint32_t End = 11;
	
	Start_timer(htim2);
	model<float> myModel(voidPtr_fixed, voidPtr_trainable, ADAM, 0.01);
	
	myModel.Init();
	//myModel.loadWeights();
	myModel.loadWeights(Start, End);
	
	std::vector<MnistImage> batch;
	for (int i = 0; i < 2; i++)
	{
		batch.emplace_back(MnistImage());
	}
	
	load_mnist_small(batch);
	
	const MnistImage* mnist_four_byte = &batch[0];
	static const float* mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
	
	const int numOutputs = 10;
	float output[numOutputs] = {0.0f};
	
	float expectedOutput[numOutputs] = {0.0f};
	
	int testsize = 2;
	float lossbefore = 0.0;
	float accuracy = 0.0;
	
	int trainingEpochs = 10;
	int trainsize = 2;
	int batch_size = 2;
	Protocol instruction = Protocol::None;
	
	uint32_t batchsize = 0;
	int32_t bytes_received = 0;
	uint32_t length;
	uint32_t length_to_recv = 0;
	int32_t retval = 0;
	Layerinformation_t layer_info;
	uint32_t ticks = 0;
	
	while (1)
	{
		if (instruction != Protocol::Start_Training)
		{
			// wait for new Instruction
			bytes_received = recv_from_direct(IPs[0],reinterpret_cast<uint8_t*>(&instruction),sizeof(Protocol));
		
			if (bytes_received < 0)
			{
				while(1);
			}
		
			if (bytes_received == 0)
			{
				continue;
			}
		}
		
		switch (instruction)
		{
			case Protocol::Train:
				// called from myModel.Train from the prev MC
				bytes_received = 0;
				while (bytes_received != sizeof(uint32_t))
				{
					bytes_received = recv_from_direct(IPs[0],reinterpret_cast<uint8_t*>(&length),sizeof(uint32_t));
					if (bytes_received < 0)
					{
						while(1);
					}
				}

				if (length > BUFFER_SIZE)
				{
					while(1);
				}
				
				// wait for input weigths
				bytes_received = 0;
				while (bytes_received != length)
				{
					bytes_received = recv_from_direct(IPs[0],buffer,length);
					if (bytes_received < 0)
					{
						while(1);
					}
				}
				
				// wait for expectedOutput
				bytes_received = 0;
				while (bytes_received != myModel.mExpectedOutputSize * sizeof(uint32_t))
				{
					bytes_received = recv_from_direct(IPs[0],reinterpret_cast<uint8_t*>(expectedOutput),myModel.mExpectedOutputSize * sizeof(uint32_t));
					if (bytes_received < 0)
					{
						while(1);
					}
				}

				myModel.Train(reinterpret_cast<float*>(buffer), expectedOutput, Start, End, &CommunicationData);
				break;
				
			case Protocol::Update:
				// get batchsize
				bytes_received = 0;
				while (bytes_received != sizeof(uint32_t))
				{
					bytes_received = recv_from_direct(IPs[0],reinterpret_cast<uint8_t*>(&batchsize),sizeof(uint32_t));
					if (bytes_received < 0)
					{
						while(1);
					}
				}
			
				if (batchsize == 0)
				{
					while(1);
				}
				
				// make a Update from the local Gradients
				myModel.Update(batchsize, Start, End);
				myModel.deleteGradients(Start, End);
				myModel.initGradients(Start, End);
				break;
				
			case Protocol::Safe_and_Swap:
				ticks = Stop_timer(htim2);
				Start_timer(htim2);
				// clear flash from old Data
				fm.EraseFlash(start_data_32,SectorAmount,(uint32_t*)SectorNumbers);
			
				// save Weights in Flash
				myModel.saveWeights(&fm, (uint32_t*)start_data, Start, End);
			
				myModel.UpdateFlash(CommunicationData, buffer, BUFFER_SIZE, fm, (uint32_t*)start_data, Start, End);
			
				ticks = Stop_timer(htim2);
				
				break;
			
			case Protocol::Start_Training:
				for (int epoch = 0; epoch < trainingEpochs; epoch++)
				{
					for (int x = 0; x < trainsize / batch_size; x++)
					{
						myModel.initGradients();
						for (int i = 0; i < batch_size; i++)
						{
							mnist_four_byte = &batch[x*batch_size + i];
							mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
							expectedOutput[mnist_four_byte->label] = 1.0;
							//myModel.Train(mnist_test_data, expectedOutput);
							myModel.Train(mnist_test_data, expectedOutput, Start, End, &CommunicationData);
							expectedOutput[mnist_four_byte->label] = 0.0;
						}
						
						// send all devices the Update command
						instruction = Protocol::Update;
						broadcast_direct(reinterpret_cast<uint8_t*>(&instruction),sizeof(Protocol));
						
						// send batchsize
						broadcast_direct(reinterpret_cast<uint8_t*>(&batch_size),sizeof(batch_size));
						
						//myModel.Update(batch_size);
						//myModel.deleteGradients();
						
						myModel.Update(batch_size, Start, End);
						myModel.deleteGradients(Start, End);
						
						// clear local Weigths and Bias
						instruction = Protocol::Safe_and_Swap;
						broadcast_direct(reinterpret_cast<uint8_t*>(&instruction),sizeof(Protocol));
						
						//fm.EraseFlash(start_data_32, SectorAmount, (uint32_t*)SectorNumbers);
						
						//myModel.saveWeights(&fm, (uint32_t*)start_data, Start, Stop);
						
						//myModel.UpdateFlash(CommunicationData, buffer, BUFFER_SIZE, fm, (uint32_t*) start_data, Start, Stop, true);
					}
				}
				ticks = Stop_timer(htim2);
				break;
			
			case Protocol::None:
				// Used when no message it recived
				break;
			case Protocol::Inference:
				for	(int x = 0; x < testsize; x++)
				{
					mnist_four_byte = &batch[x];
					mnist_test_data = reinterpret_cast<const float*>(mnist_four_byte->data);
					myModel.InferenceSRAM(mnist_test_data, output);
					
					expectedOutput[mnist_four_byte->label] = 1.0;
					
					for (int i = 0; i < numOutputs; i++)
					{
						if (expectedOutput[i] > 0.0f)
						{
							lossbefore -= std::log(output[i] + 1e-9f);
						}
					}
					
					expectedOutput[mnist_four_byte->label] = 0.0;
				}
				break;
			default:
				while(1)
				{
					// Unknown message
				}
				break;
		}
		
		// reset instruction
		instruction = Protocol::None;
	}
	
  while (1)
  {
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
