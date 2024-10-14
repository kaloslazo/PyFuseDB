from emoji import emojize
from termcolor import colored
from DataLoader import DataLoader
from SqlParser import SqlParser
from GradioUI import createDemo

def main():
    print(emojize(f"\n:snake: {colored("PyFuseDB: Sistema de Recuperación de Información", "green")}"))

    print(emojize(f"\n:file_cabinet: {colored("Cargando el dataset FMA: A Dataset For Music Analysis", "blue")}"))
    dataLoader = DataLoader("data/fma_small.csv")
    dataLoader.loadData()

    print(emojize(f"\n:brain: {colored("PyFuseDB: Preparando parser SQL", "blue")}"))
    sqlParser = SqlParser()

    print(emojize(f"\n:rocket: {colored("Inicializando UI", "blue")}"))
    demo = createDemo(dataLoader, sqlParser)
    demo.launch()

if __name__ == '__main__':
    main()
