# Universidad del Valle de Guatemala
# Grafica por Computadora
# Nombre: Marcos Gutierrez
# Carne: 17909

from gl import *
import random

r = Render(800,800)
r.glCreateWindow(800,800)
r.glViewPort(0,0,999,999)

t = Texture('model.bmp')

# Rotacion a la Derecha
r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(0,1,0), texture=t)
r.archivo('Derecha.bmp')

# Rotacion Izquierda
#r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
#r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(0,-1,0), texture=t)
#r.archivo('Izquierda.bmp')

# Arriba
#r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
#r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(1,0,0), texture=t)
#r.archivo('Arriba.bmp')

# Abajo
#r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
#r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(-0.5,0,0), texture=t)
#r.archivo('Abajo.bmp')
