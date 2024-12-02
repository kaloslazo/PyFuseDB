from emoji import emojize
from termcolor import colored
from GradioUI import createDemo


if __name__ == '__main__':
    print(emojize(f'\n:snake: {colored("PyFuseDB: Sistema de Recuperación de Información", "green")}'))
    print(emojize(f'\n:rocket: {colored("Inicializando UI", "blue")}'))
    
    demo = createDemo()
    demo.queue()
    demo.launch()
