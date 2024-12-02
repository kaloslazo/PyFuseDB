![](./docs/banner.png)

**PyFuseDB** es un sistema que integra varios modelos de datos y técnicas avanzadas de recuperación de información dentro de una única base de datos. Nuestro sistema permite a los usuarios recuperar datos estructurados mediante un **índice invertido** y datos no estructurados, como imágenes y audio, por medio de **estructuras multidimensionales** que utilizan vectores caracterísicos.

> 📚 Para obtener más detalles sobre el proyecto PyFuseDB, te recomendamos visitar la [wiki](https://github.com/kaloslazo/PyFuseDB/wiki), donde encontrarás información adicional y documentación más detallada.

## Setup

1. Crear un environment de conda:
```bash
$ conda env create -f pyfuse.yaml
$ conda activate pyfuse
```

2. Descargar las dependencias en zip al root del repositorio [pyfuse-dependencies.zip](https://drive.google.com/file/d/1JRl4dTjymoYs_7hPuOerGOmSyvbSlTZm/view?usp=sharing)
```bash
$ unzip pyfuse-dependencies.zip
```

3. Entrar a la carpeta `app/` y ejecutar
```bash
$ cd app
$ python main.py
```
