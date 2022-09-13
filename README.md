**Trabajo Fin  de Master. Máster Universitario Oficial en Ciencia de Datos e Ingeniería de Computadores**

**José Miguel Ibáñez Mengual**

Curso 2021-2022

Universidad de Granada

## Segmentación semántica de objetos celestes en imágenes astronómicas usando Deep Learning 
### Resumen
Recientemente se han publicado las primeras imágenes obtenidas con el
telescopio espacial James Webb de la NASA lanzado en diciembre de 2021. Se
ha obtenido la imagen infrarroja más profunda y nítida del universo lejano hasta
la fecha, el cúmulo de galaxias SMACS 0723 tal y como lucía hace 4.600 millones
de años. Dicha imagen está repleta de miles de galaxias, incluidos los objetos
más tenues jamás observados en el infrarrojo. Las galaxias observadas tienen
estructuras diminutas y tenues que nunca antes habían sido observadas,
incluidos cúmulos de estrellas y características difusas.
La clasificación morfológica de las galaxias es una tarea clave para
entender la formación y evolución de las galaxias. Con la nueva instrumentación
astronómica, el volumen de datos generado por telescopios espaciales y
terrestres es cada vez mayor, lo que hace cada vez más necesario el disponer
de técnicas avanzadas de visión por computadora para procesar el gran
volumen de datos. Una de las tareas fundamentales es la de detectar y clasificar
los objetos celestes que aparecen en las imágenes astronómicas, como son las
galaxias. A pesar del enorme avance producido en los últimos años en los
modelos del aprendizaje profundo y su aplicación a la visión por computadora,
su implicación en el campo de la detección, segmentación y clasificación
morfológica de imágenes astronómica es aún escaso.

En este trabajo vamos a abordar la revisión y evaluación de las algunas
técnicas del estado del arte para la segmentación de imágenes usando
modelos basados en aprendizaje profundo. Mostraremos especial atención a
algunas propuestas para mejorar los problemas y desafíos actuales, como, por
ejemplo, la falta de datos etiquetados y la aplicación de técnicas semisupervisadas. 
En primer lugar, revisaremos las técnicas clásicas y del estado del
arte segmentación de imágenes basada aprendizaje profundo, haciendo una
breve introducción y clasificación de los principales modelos de aprendizaje
profundo. En segundo lugar, repasaremos los principales métodos usados
actualmente para la de segmentación de imágenes astronómicas y su
aplicación a la clasificación morfológica de galaxias. Posteriormente,
aplicaremos algunas de las técnicas anteriores a un conjunto de datos de
imágenes de galaxias generado a partir de las bases de datos disponibles de
forma abierta del proyecto de ciencia ciudadana Galaxy Zoo. Mostraremos los
resultados, comparemos las distintas técnicas discutiendo las fortalezas y
debilidades de cada una y su potencial aplicación en el campo de las
imágenes astronómicas. Finalmente, discutiremos los principales desafíos y
trabajos futuros derivados del estudio y de los resultados obtenidos en presente
trabajo.


This repo contains implementations of different semantic segmentation models for a Galaxy Zoo2 based image catalog.

* Dataset
   - [Galaxy Zoo](https://data.galaxyzoo.org/) 
   - [Tools](https://github.com/jm-ibanez/TFM/tree/main/Dataset) 
* Supervised segmentation
   - Based on [Detectron2](https://github.com/facebookresearch/detectron2)
   - Based on [Pytorch](https://github.com/yassouali/pytorch-segmentation) 
* Semi-Supervised segmentation
   - [DMT](https://github.com/voldemortX/DST-CBC/blob/master/README.md)
   - [USSS](https://github.com/tarun005/USSS_ICCV19)
* Models
   - [DMT](https://tensorboard.dev/experiment/djB0GHj0QcOm5MhbFGoAXg/#scalars)
* Document
* [Demo1](https://colab.research.google.com/drive/1iHoWRDG8qbotYCECG88tN57eZcu6c8YY?usp=sharing)
