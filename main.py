# SR6-Camaras
# Graficas por computadora 
# Esteban Aldana Guerra 20591

from gl import *
import random

r = Render(800,800)
r.glCreateWindow(800,800)
r.glViewPort(0,0,999,999)

t = Texture('coca.bmp')

# Rotacion a la Derecha
r.lookAt(V3(-0.2,0,20), V3(0,0,0), V3(0,1,0))
r.load('cocacola.obj',translate=(-1,-1,0), scale=(0.1,0.1,0.1), rotate=(0,0.2,0.36), texture=t)
r.archivo('coquita.bmp')

# Rotacion Izquierda
#r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
#r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(0,-1,0), texture=t)
#r.archivo('Izquierda.bmp')

# Arriba
#r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
#r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(1,0,0), texture=t)
#r.archivo('Arriba.bmp')

# Low
#r.lookAt(V3(1,0,5), V3(0,0,0), V3(0,1,0))
#r.load('model.obj',translate=(0,0,0), scale=(0.5,0.5,0.5), rotate=(-0.5,0,0), texture=t)
#r.archivo('Low.bmp')
