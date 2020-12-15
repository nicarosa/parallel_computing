#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>

#define MSG_LENGTH 10

using namespace cv;
const int alto = 480;
const int ancho = 854;

int main(int argc, char *argv[])
{
  int tareas, val;
  //Manejo de tiempo
  double time, execTime;
  
  //Direccion de imagen de entrada
  std::string image_in_path = argv[1];
  //Direccion de imagen de salida
  std::string image_out_path = argv[2];
 
  //Manejo de imagen y captura
  Mat image = imread(image_in_path, IMREAD_COLOR);
  if (image.empty())
  {
    std::cout << "No se pudo leer la imagen: " << image_in_path << std::endl;
    return 1;
  }
  
  uint8_t *pixelPtr = (uint8_t *)image.data;
  int canales = image.channels(), columnas = image.cols, filas = image.rows;
  uint8_t *resized = (uint8_t *)malloc(image.channels() * alto * ancho * sizeof(uint8_t));
  //Fin de manejo de imagen y captura
  
  //Estado de la operacion de recepcion
  MPI_Status status; 
  //Inicializa el entorno de ejecucion de mpi
  MPI_Init(&argc, &argv); 

  //Variables de tiempo
  double t1, t2;

  //Determina el tamanio del grupo comm
  MPI_Comm_size(MPI_COMM_WORLD, &tareas);
  //Determina el rango del proceso llamante en el comm
  MPI_Comm_rank(MPI_COMM_WORLD, &val);

  //Envia el mensaje de val
  MPI_Bcast(pixelPtr, canales * columnas * filas, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  //Bloquea hasta que se haya recivido la rutina
  MPI_Barrier(MPI_COMM_WORLD);

  int split = alto / tareas;
  int filaP = (val * split);

  uint8_t *_resized = (uint8_t *)malloc(canales * split * ancho * sizeof(uint8_t));
  
  // Inicio del algoritmo de reduccion
  // Definicion de T1
  t1 = MPI_Wtime();    

  float radio_x = (columnas - 1) / (ancho - 1);
  float radio_y = (filas - 1) / (alto - 1);
  uint8_t cUno, cDos, cTres, cQuatro, pixel;
  int cnter = 0;
  for(int i = filaP; i < min(filaP+split, alto); i++)
  {
    for (int j = 0; j < ancho; j++)
    {

      int x_l = floor(radio_x * j), y_l = floor(radio_y * i);
      int x_h = ceil(radio_x * j), y_h = ceil(radio_y * i);

      float peso_x = (radio_x * j) - x_l;
      float peso_y = (radio_y * i) - y_l;

      for (int k = 0; k < canales; k++)
      {
        cUno = pixelPtr[y_l * columnas * canales + x_l * canales + k];
        cDos = pixelPtr[y_l * columnas * canales + x_h * canales + k];
        cTres = pixelPtr[y_h * columnas * canales + x_l * canales + k];
        cQuatro = pixelPtr[y_h * columnas * canales + x_h * canales + k];

        pixel = (cUno & 0xff) * (1 - peso_x) * (1 - peso_y) + (cDos & 0xff) * peso_x * (1 - peso_y) + (cTres & 0xff) * peso_y * (1 - peso_x) + (cQuatro & 0xff) * peso_x * peso_y;       
        _resized[(cnter) * ancho * canales + j * canales + k] = pixel;
      }      
    }
    cnter++;
  }
  
  //Definicion de T2
  t2 = MPI_Wtime();   
  // Fin del algoritmo de reduccion 

  //Calculo de tiempo
  execTime = t2 - t1;

  //Calculo de reduccion
  MPI_Reduce(&execTime, &time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  //Union de elementos
  MPI_Gather(_resized, alto * ancho * canales / tareas, MPI_UINT8_T, resized, alto * ancho * canales / tareas, MPI_UINT8_T, 0, MPI_COMM_WORLD);

  //Termina entorno de ejecucio MPI
  MPI_Finalize();
  
  //Toma de tiempos y escritura de la imagen de salida
      if (val == 0)
  {
    printf("%d,%1.5f\n",tareas, time);
    Mat resized_image(alto, ancho, CV_8UC(3), resized);
    imwrite(image_out_path, resized_image);
  }
}
