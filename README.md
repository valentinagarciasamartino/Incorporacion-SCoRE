# Incorporacion-SCoRE
Aquí se encuentran los códigos utilizados para llevar a cabo el proyecto final integrador titulado "INCORPORACIÓN DEL MÉTODO FUNCIONAL SCoRE"

Los códigos se pueden dividir en 3 grupos.

1) Códigos utilizados para la implementación computacional del método
   - score_hjc.py: archivo principal del programa, encargado de coordinar el flujo completo de procesamiento y ejecutar el cálculo del HJC.
   - extraFunctions.py: archivo auxiliar que implementa los algoritmos biomecánicos principales, incluyendo OCST (Optimal Common Shape Technique), y SCoRE.
   - Add_marker.py: función auxiliar utilizada para escribir los nuevos marcadores calculados dentro de un archivo .tdf existente. 
   - score.cfg: archivo de configuración que contiene los parámetros que pueden variar entre protocolos de captura, como los nombres de marcadores, nombres de archivos y configuraciones del filtro e interpolación. Tiene la ventaja de que se puede modificar este archivo y no todo el código. 
   - score_config.py: archivo auxiliar que carga score.cfg y expone sus valores como propiedades de Python, para que el resto de los scripts no tengan que leer el archivo de configuración directamente. 

2) Códigos utilizados para la integración del método al entorno del laboratorio
   - Processor.py: es el encargado de gestionar el procesamiento de los datos del paciente.
   - ProcessorCli.py: a través de su CLI, el usuario puede ejecutar el comando procesar carpeta, que automatiza una serie de pasos: el etiquetado de marcadores, la marcación de eventos y la aplicación del protocolo de HHMM.
   - Reporter.py: es el encargado de generar el informe PDF final del análisis. 
   - ReporterCli.py: a través de su CLI, toma los archivos MDX procesados y produce las páginas del informe con los gráficos cinemáticos, cinéticos, de longitud muscular y de electromiografía.

3) Códigos utilizados para el análisis de los resultados obtenidos
   - CoordenadaHJC.py: script utilizado para comparar las coordenadas del HJC entre métodos. 
   - CurvasCinematicas.py: script utilizado para comparar el efecto del HJC obtenido sobre las curvas cinemáticas de la marcha.
   
