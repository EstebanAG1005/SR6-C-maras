# SR5-Textures
# Graficas por computadora 
# Esteban Aldana Guerra 20591

from gl import Render, color, V2, V3
from textures import Obj, Texture

import random

r = Render(1300, 1300)
r.glCreateWindow(1300,1300)
r.glViewPort(0,0,999,999)

t = Texture('model.bmp')

r.lookAt(V3(1,0,5), V3(-0.3,-0.3,0), V3(0,1,0))
r.load('model.obj', (1.5, 1.5, 1), (400, 400, 400), rotate=(0,1,0), texture=t)

r.archivo('output.bmp')