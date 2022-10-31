/*-----------------------------------------------------------------------------
 Crazyflie drone STM32 deployment of algorithm for autonomous flight. Code to build and flash on STM32

 Angel Canelo 2022.10.18

 Code modified from Bitcraze crazyflie-firmware app_hello_world                                                  
-------------------------------------------------------------------------------*/

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"
#include "debug.h"
#include "uart_dma_setup.h"
#include "commander.h"
#include "log.h"
#include "param.h"

#define DEBUG_MODULE "HELLOWORLD"
#define BUFFERSIZE 1

uint8_t aideckRxBuffer[BUFFERSIZE];
volatile uint8_t dma_flag = 0;
uint8_t log_counter=0;
int result[5];	// array of 5 elements
int elem;
float chtime;
float chtime2;

static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;


  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;

  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;

  setpoint->velocity_body = true;
}

typedef enum {
    idle,
    //lowUnlock,
    unlocked,
    stopping
} State;

static State state = idle;

static const float height_sp = 0.2f;
static float height_sp2 = 0.3f;

#define MAX(a,b) ((a>b)?a:b)
#define MIN(a,b) ((a<b)?a:b)

int get_most_common(int vet[], size_t dim)
{
    size_t i, j, count;
    size_t most = 0;
    int temp;

    for(i = 0; i < dim; i++) {
        temp = vet[i];
        count = 1;
        for(j = i + 1; j < dim; j++) {
            if(vet[j] == temp) {
                count++;
            }
        }
        if (most < count) {
            most = count;
            elem = vet[i];
        }
    }
    return elem;
}

void appMain()
{

	static setpoint_t setpoint;

  	vTaskDelay(M2T(10000));  
  	paramVarId_t idPositioningDeck = paramGetVarId("deck", "bcFlow2");

	USART_DMA_Start(115200, aideckRxBuffer, BUFFERSIZE);
	int count = 0;
	int finres = 3;
	int ddcheck = 0;
	while(1) {
		vTaskDelay(M2T(10));
		uint8_t positioningInit = paramGetUint(idPositioningDeck);
		if (state == unlocked)
		{
			if (finres==3 && usecTimestamp()/1000000<=chtime2+5) {	// Initial hovering
				setHoverSetpoint(&setpoint, 0, 0, height_sp, 0);
				commanderSetSetpoint(&setpoint, 3);				
			}
			else if (finres==3 && usecTimestamp()/1000000>chtime2+5){
				setHoverSetpoint(&setpoint, 0, 0, height_sp2, 0);
				commanderSetSetpoint(&setpoint, 3);
				if (usecTimestamp()/1000000>=chtime2+8){
					finres = 4;
				}
			}
			else {
				if (finres==4) {	// Moving fordward
					setHoverSetpoint(&setpoint, 0.1f, 0, height_sp2, 0);
					commanderSetSetpoint(&setpoint, 3);
					if (dma_flag == 1) {
						dma_flag = 0;  // clear the flag
						result[count] = aideckRxBuffer[0];
						if (count==5){
							finres = get_most_common(result, 5);
							count = 0;
							if (finres==0)
							{
								DEBUG_PRINT("Collision: %d\n", finres);
							}
							else if (finres==1)
							{
								DEBUG_PRINT("Rectangle: %d\n", finres);
								chtime = usecTimestamp()/1000000;
							}
							else if (finres==2)
							{
								DEBUG_PRINT("Square: %d\n", finres);
								chtime = usecTimestamp()/1000000;
							}
						}
						log_counter = aideckRxBuffer[0];
						memset(aideckRxBuffer, 0, BUFFERSIZE);  // clear the dma buffer
						count = count + 1;
					}
				}
				if (finres==0) {
					setHoverSetpoint(&setpoint, 0, 0, 0.1f, 0);
					commanderSetSetpoint(&setpoint, 3);
					state = stopping;
				}
				if (finres==1 && usecTimestamp()/1000000<=chtime+2.5f) {	// rotation right 30deg/s (90 deg)
					setHoverSetpoint(&setpoint, 0, 0, height_sp2, -30.0f);
					commanderSetSetpoint(&setpoint, 3);
				}
				else if (finres==1 && usecTimestamp()/1000000>chtime+2.5f){
					finres = 4;
				}
				if (finres==2 && usecTimestamp()/1000000<=chtime+2.5f) {	// rotation left 30deg/s (-90 deg)
					setHoverSetpoint(&setpoint, 0, 0, height_sp2, 30.0f);
					commanderSetSetpoint(&setpoint, 3);
				}
				else if (finres==2 && usecTimestamp()/1000000>chtime+2.5f){
					finres = 4;
				}			
			}
		}
		///// Check state conditions
		else{

			if (positioningInit && state == idle)
			{
				DEBUG_PRINT("Unlocked!\n");
				state = unlocked;
				chtime2 = usecTimestamp()/1000000;
			}
			if (state == stopping)
			{
				memset(&setpoint, 0, sizeof(setpoint_t));
				commanderSetSetpoint(&setpoint, 3);
				DEBUG_PRINT("Landed\n");
			}
		}
	}
}


void __attribute__((used)) DMA1_Stream1_IRQHandler(void)
{
 DMA_ClearFlag(DMA1_Stream1, UART3_RX_DMA_ALL_FLAGS);
 dma_flag = 1;
}

LOG_GROUP_START(log_test)
LOG_ADD(LOG_UINT8, test_variable_x, &log_counter)
LOG_GROUP_STOP(log_test)
