# Universidad del Valle de Guatemala
# Grafica por Computadora
# Nombre: Marcos Gutierrez
# Carne: 17909

from gl import *
import random

objetos = Render(800,800)
objetos.glCreateWindow(800,800)
objetos.glViewPort(0,0,999,999)

# TOQUE DE LA CASA
# En el siguiente codigo se renderizara a goku
#Codigo para renderizar a goku
goku = Texture('model.bmp')
objetos.lookAt(V3(1,0,5), V3(-0.3,-0.3,0), V3(0,1,0))
objetos.load('model.obj',translate=(0,0,0), scale=(1,1,1), rotate=(0,1,0), texture=goku)

#Importamos la funcion donde se imprimira todo
objetos.archivo('Prueba.bmp')
