# Incorporación del método funcional SCoRE

Repositorio correspondiente al Proyecto Final Integrador “Incorporación del método funcional SCoRE en la dinámica operativa de un laboratorio de análisis de marcha”. El proyecto tiene como objetivo incorporar el método funcional SCoRE (Symmetrical Center of Rotation Estimation) para la estimación del centro de rotación de la cadera (HJC, Hip Joint Center) al flujo de trabajo de un laboratorio de análisis de marcha.

El código del proyecto se organiza en tres grupos principales: implementación computacional del método SCoRE, integración al entorno del laboratorio y análisis de los resultados obtenidos.

## 1. Implementación del método SCoRE

La carpeta `score/` contiene los archivos correspondientes a la implementación computacional del método.

- score_hjc.py: archivo principal del programa. Coordina el flujo de procesamiento y ejecuta el cálculo del HJC mediante el método SCoRE.
- extraFunctions.py: contiene funciones auxiliares y los algoritmos biomecánicos utilizados, incluyendo OCST (Optimal Common Shape Technique) y SCoRE.
- add_marker.py: función auxiliar utilizada para escribir los nuevos marcadores calculados dentro de un archivo `.tdf`.
- score.cfg: archivo de configuración que contiene parámetros que pueden variar entre protocolos, como nombres de marcadores, nombres de archivos y parámetros de filtrado e interpolación.
- score_config.py: carga el archivo `score.cfg` y permite acceder a sus parámetros desde Python.

El cálculo utiliza una calibración estática y una calibración funcional de cada lado. A partir de las trayectorias de los clústeres de pelvis y muslo se estiman las transformaciones de cuerpo rígido y posteriormente se aplica SCoRE para obtener el HJC. El resultado se transforma al sistema de referencia local de la pelvis y se escribe como un nuevo marcador en una copia del archivo standing.

## 2. Integración al entorno del laboratorio

La carpeta `integración/` contiene los scripts utilizados para incorporar SCoRE al flujo de procesamiento existente del laboratorio.

- Processor.py: gestiona el procesamiento de los datos del paciente.
- ProcessorCli.py: interfaz de línea de comandos utilizada para automatizar el procesamiento de las capturas e incorporar la ejecución del protocolo SCoRE.
- Reporter.py: genera las distintas secciones del informe del análisis de marcha e incorpora los resultados obtenidos utilizando SCoRE.
- ReporterCli.py: interfaz de línea de comandos utilizada para generar los informes a partir de los archivos procesados.
- Figures.py: contiene las funciones y clases utilizadas para generar las figuras de los informes.
- functions.py: contiene funciones auxiliares utilizadas durante la generación y procesamiento de los resultados.
- Consist.py: contiene funciones utilizadas por `ReporterCli.py` para el procesamiento y generación de informes.
- copy_tdf.py: realiza la preparación/copia del archivo TDF necesaria antes de ejecutar el cálculo de SCoRE.

## 3. Análisis de resultados

La carpeta `análisis/` contiene los scripts utilizados para analizar los resultados obtenidos durante el desarrollo del proyecto.

- CoordenadaHJC.py: compara las coordenadas del HJC obtenidas mediante diferentes métodos.
- CurvasCinematicas.py: permite analizar y comparar el efecto de la ubicación del HJC sobre las curvas cinemáticas de la marcha.

## Requisitos

El proyecto fue desarrollado para ejecutarse con Python 3.8 o superior. Las dependencias de Python necesarias se encuentran especificadas en `requirements.txt` y pueden instalarse mediante:

```bash
pip install -r requirements.txt
```

## Configuración de SCoRE

Los parámetros utilizados por el cálculo se encuentran en .../score/score.cfg. Este archivo permite modificar nombres de marcadores, archivos de entrada y parámetros de procesamiento sin modificar directamente el código fuente.

## Consideraciones sobre el entorno del laboratorio

Los scripts contenidos en `integracion/` fueron desarrollados a partir del flujo de procesamiento utilizado en el laboratorio y, por lo tanto, algunas funciones dependen del entorno de software y de la estructura de archivos allí utilizada. Los recursos institucionales utilizados para la generación de informes, como templates, imágenes, logos y fuentes, no se incluyen en este repositorio.

Por este motivo, la generación completa de los informes mediante `Reporter.py` puede requerir recursos adicionales disponibles únicamente en el entorno del laboratorio. Asimismo, algunos scripts conservan rutas y configuraciones propias de dicho entorno, que deben adaptarse en caso de utilizar el código en otra instalación.

## Datos

Los archivos correspondientes a capturas y datos de pacientes no forman parte de este repositorio. El archivo `.gitignore` está configurado para evitar incorporar accidentalmente al repositorio archivos de datos y resultados generados durante el procesamiento.

## Autor

Valentina Garcia Samartino  
Proyecto Final Integrador — Ingeniería Biomédica - UNSAM
   
