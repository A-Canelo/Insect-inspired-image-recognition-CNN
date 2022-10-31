/*-----------------------------------------------------------------------------
 Crazyflie AI-deck drone deployment of FlyDrosNet CNN for image recognition. Code to build and flash on AI-deck GAP8

 Angel Canelo 2022.10.18

 Code modified from Bitcraze framework of ai-deck ai classification example                                                   
-------------------------------------------------------------------------------*/

#include "classification.h"
#include "bsp/camera.h"
#include "bsp/camera/himax.h"
#include "bsp/buffer.h"
#include "bsp/transport/nina_w10.h"
#include "classificationKernels.h"
#include "gaplib/ImgIO.h"
#include "pmsis.h"
#include "stdio.h"
#include "bsp/bsp.h"
#include "cpx.h"

#define CAM_WIDTH 324
#define CAM_HEIGHT 244

#define CHANNELS 1
#define IO RGB888_IO
#define CAT_LEN sizeof(uint32_t)

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s

static pi_task_t task1;
static pi_task_t task2;
static unsigned char *cameraBuffer;
static signed short *Output_1;
static struct pi_device camera;
static pi_buffer_t buffer;
static EventGroupHandle_t evGroup;
#define CAPTURE_DONE_BIT (1 << 0)
static struct pi_device cluster_dev;
static struct pi_cluster_task *task;
static struct pi_cluster_conf conf;


static uint8_t to_send;

AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;

#define IMG_ORIENTATION 0x0101


static void RunNetwork()
{
  __PREFIX(CNN)
  (cameraBuffer, Output_1);
}

static void cam_handler(void *arg)
{
  xEventGroupSetBits(evGroup, CAPTURE_DONE_BIT);
}

static int open_camera(struct pi_device *device)
{

  struct pi_himax_conf cam_conf;

  pi_himax_conf_init(&cam_conf);

  cam_conf.format = PI_CAMERA_QVGA;

  pi_open_from_conf(device, &cam_conf);
  if (pi_camera_open(device))
    return -1;
  pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
  uint8_t set_value = 3;
  uint8_t reg_value;
  pi_camera_reg_set(&camera, IMG_ORIENTATION, &set_value);
  pi_time_wait_us(1000000);
  pi_camera_reg_get(&camera, IMG_ORIENTATION, &reg_value);

  if (set_value != reg_value)
  {
    return -1;
  }
              
  pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);

  pi_camera_control(device, PI_CAMERA_CMD_AEG_INIT, 0);
  return 0;
}

// Functions and init for LED toggle
#define LED_PIN 2
static pi_device_t led_gpio_dev;
void hb_task(void *parameters)
{
  (void)parameters;
  char *taskname = pcTaskGetName(NULL);

  // Initialize the LED pin
  pi_gpio_pin_configure(&led_gpio_dev, LED_PIN, PI_GPIO_OUTPUT);

  const TickType_t xDelay = 500 / portTICK_PERIOD_MS;

  while (1)
  {
    pi_gpio_pin_write(&led_gpio_dev, LED_PIN, 1);
    vTaskDelay(xDelay);
    pi_gpio_pin_write(&led_gpio_dev, LED_PIN, 0);
    vTaskDelay(xDelay);
  }
}

int classification()
{
  uint32_t errors = 0;

  struct pi_device uart;
  struct pi_uart_conf conf2;
  /* Init & open uart. */
  pi_uart_conf_init(&conf2);
  conf2.enable_tx = 1;
  conf2.enable_rx = 0;
  conf2.baudrate_bps = 115200;
  pi_open_from_conf(&uart, &conf2);
  if (pi_uart_open(&uart))
  {
      pmsis_exit(-1);
  }

  evGroup = xEventGroupCreate();

    // Start LED toggle
  BaseType_t xTask;
  xTask = xTaskCreate(hb_task, "hb_task", configMINIMAL_STACK_SIZE * 2,
                      NULL, tskIDLE_PRIORITY + 1, NULL);

  if (open_camera(&camera))
  {
    pmsis_exit(-1);
  }
  uint32_t resolution = CAM_WIDTH * CAM_HEIGHT;
  uint32_t captureSize = resolution * sizeof(unsigned char);
  cameraBuffer = (unsigned char *)pmsis_l2_malloc(captureSize);
  if (cameraBuffer == NULL)
  {
    return -1;
  }
  pi_buffer_init(&buffer, PI_BUFFER_TYPE_L2, cameraBuffer);
  pi_buffer_set_format(&buffer, CAM_WIDTH, CAM_HEIGHT, 1, PI_BUFFER_FORMAT_GRAY);

  Output_1 = (signed short *)pmsis_l2_malloc(3 * sizeof(signed short));
  if (Output_1 == NULL)
  {
    pmsis_exit(-1);
  }

  /* Configure CNN task */
  pi_cluster_conf_init(&conf);
  pi_open_from_conf(&cluster_dev, (void *)&conf);
  pi_cluster_open(&cluster_dev);
  task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));

  memset(task, 0, sizeof(struct pi_cluster_task));
  task->entry = &RunNetwork;
  task->stack_size = STACK_SIZE;             // defined in makefile
  task->slave_stack_size = SLAVE_STACK_SIZE; // "
  task->arg = NULL;

  /* Construct CNN */
  if (__PREFIX(CNN_Construct)())
  {
    pmsis_exit(-5);
  }
  pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
  while(1)
  {
  pi_camera_capture_async(&camera, cameraBuffer, resolution, pi_task_callback(&task1, cam_handler, NULL));
  pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
  xEventGroupWaitBits(evGroup, CAPTURE_DONE_BIT, pdTRUE, pdFALSE, (TickType_t)portMAX_DELAY);
  pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
  
    /* Run inference */
  pi_cluster_send_task_to_cl(&cluster_dev, task);

  if (Output_1[0] > Output_1[1] && Output_1[0] > Output_1[2])
  {
    uint8_t sen=0;
    to_send = sen;
  }
  else if (Output_1[1] > Output_1[0] && Output_1[1] > Output_1[2])
  {
    uint8_t sen=1;
    to_send = sen;
  }
  else if (Output_1[2] > Output_1[0] && Output_1[2] > Output_1[1])
  {
    uint8_t sen=2;
    to_send = sen;
  }
    pi_uart_write(&uart, &to_send, 1);
    pi_time_wait_us(500000);

  }
  pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
  pi_camera_close(&camera);
  pi_uart_close(&uart);
  
  /* Destruct CNN */
  __PREFIX(CNN_Destruct)
  ();
  pmsis_exit(0);
  return 0;
}

int main(void)
{  
  pi_bsp_init();

  // Increase the FC freq to 250 MHz
  pi_freq_set(PI_FREQ_DOMAIN_FC, 250000000);
  pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
  return pmsis_kickoff((void *)classification);
}