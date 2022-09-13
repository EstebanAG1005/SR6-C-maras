# SR5-Textures
# Graficas por computadora 
# Esteban Aldana Guerra 20591

import struct
import random
from textures import Texture, Obj
from collections import namedtuple
from math import cos, sin

V2 = namedtuple('Vertex2', ['x', 'y'])
V3 = namedtuple('Vertex3', ['x', 'y', 'z'])

# -------------------------------------------- Utils ---------------------------------------------------

# 1 byte
def char(c):
    return struct.pack('=c', c.encode('ascii'))

# 2 bytes
def word(c):
    return struct.pack('=h', c)

# 4 bytes 
def dword(c):
    return struct.pack('=l', c)

# Funcion de Color
def color(r, g, b):
    return bytes([b, g, r])

# Funcion que ayuda a convertir floats para poder usar los colores
def FloatRGB(array):
    return [round(i*255) for i in array]

# ----------------------------- Parte de Operaciones Matematicas -----------------------------------------

# Coordenadas Baricentricas
def barycentric(A, B, C, P):
    bary = cross(
        V3(C.x - A.x, B.x - A.x, A.x - P.x), 
        V3(C.y - A.y, B.y - A.y, A.y - P.y)
    )

    if abs(bary[2]) < 1:
        return -1, -1, -1  

    return (
        1 - (bary[0] + bary[1]) / bary[2], 
        bary[1] / bary[2], 
        bary[0] / bary[2]
    )

# Resta
def sub(v0, v1):
    return V3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z)

# producto punto
def dot(v0, v1):
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z

# Obtiene 2 valores de 3 vectores y devuelve un vector 3 con el producto punto
def cross(v0, v1):
    return V3(
    v0.y * v1.z - v0.z * v1.y,
    v0.z * v1.x - v0.x * v1.z,
    v0.x * v1.y - v0.y * v1.x,
)

# Regresa el largo del vector
def length(v0):
    return (v0.x**2 + v0.y**2 + v0.z**2)**0.5

# Normal del vector 
def norm(v0):
    v0length = length(v0)

    if not v0length:
        return V3(0, 0, 0)

    return V3(v0.x/v0length, v0.y/v0length, v0.z/v0length)

# 2 vectores de tamaño 2 que definen el rectángulo delimitador más pequeño posible
def bbox(*vertices): 
    xs = [ vertex.x for vertex in vertices ]
    ys = [ vertex.y for vertex in vertices ]
    xs.sort()
    ys.sort()

    return V2(xs[0], ys[0]), V2(xs[-1], ys[-1])

# ---------------------------------------------------- Se trabaja la multiplicaicon de matrices -----------------------------------------------------------

def teorema(filas, columna):
	matriz = []
	for i in range(filas):
		#Añadimos una lista vacia
		matriz.append([])
		for j in range(columna):
			matriz[-1].append(0.0)
	#Retornamos la matriz creada
	return matriz

def multiplicarMatrices(m1,m2):
	#Condicion para multiplicar matrices es que el numero de columnas de una matriz debe ser el mismo que el numero de filas en la otra matriz
	#Las matrices deben de tener la misma longitud (2x2 * 2x2).... (4x4 * 4x4)
	#Basado en: https://www.geeksforgeeks.org/c-program-multiply-two-matrices/
	"""
		MATRIZ 1			MATRIZ 2
	[ 0, 0 , 0 , 0 ]	[ 0, 0 , 0 , 0 ]
	[ 0, 0 , 0 , 0 ]	[ 0, 0 , 0 , 0 ]
	[ 0, 0 , 0 , 0 ]	[ 0, 0 , 0 , 0 ]
	[ 0, 0 , 0 , 0 ]	[ 0, 0 , 0 , 0 ]
	"""
	matrizResultante = teorema(len(m1), len(m2[0]))
	for i in range(len(m1)):
		#Creamos la matriz
		for j in range(len(m2[0])):
			for k in range(len(m2)):
				#damos valores a la matriz resultante
				matrizResultante[i][j] += m1[i][k] * m2[k][j]
	#retornmaos los resultados de la matriz
	return matrizResultante

# --------------------------------------------------------------------------------------------------------------------

# Se definen colores Negro y Blanco para nuestro programa
BLACK = color(0,0,0)
WHITE = color(1,1,1)

class Render(object):
 
 def __init__(self, width, height):
        self.width = width
        self.height = height
        self.vertexColor = color(255,255,255)
        self.clearColor = color(91,204,57)
        self.clear()
        #vertexBuffer
        #inicializamos las variables necesarias
        self.light = V3(0,0,1)
        self.activeTexture = None
        self.VertexArray = []

    #Funcion para limpiar
 def clear(self):
    self.framebuffer = [
        [color(0,0,0) for x in range(self.width)]
        for y in range(self.height)
    ]
    self.zbuffer = [[-float('inf') for x in range(self.width)] for y in range(self.height)]

#Funcion para escribir
 def write(self, filename="out.bmp"):
    f = open(filename, 'bw')
    # File header (14 bytes)
    f.write(char('B'))
    f.write(char('M'))
    f.write(dword(14 + 40 + self.width * self.height * 3))
    f.write(dword(0))
    f.write(dword(14 + 40))
    # Image header (40 bytes)
    f.write(dword(40))
    f.write(dword(self.width))
    f.write(dword(self.height))
    f.write(word(1))
    f.write(word(24))
    f.write(dword(0))
    f.write(dword(self.width * self.height * 3))
    f.write(dword(0))
    f.write(dword(0))
    f.write(dword(0))
    f.write(dword(0))
    for x in range(self.height):
        for y in range(self.width):
            f.write(self.framebuffer[x][y])
    f.close()

#Funcion que crea la ventana
 def glCreateWindow(self, width, height):
    self.width = width
    self.height = height
    self.clear()

 def glViewPort(self, x, y, width, height):
    self.ViewPort_X = x
    self.ViewPort_Y = y
    self.ViewPort_H = height
    self.ViewPort_W = width

 def glVertex(self, x, y):
    self.PortX = int((x+1) * self.ViewPort_W * (1/2) + self.ViewPort_X)
    self.PortY = int((y+1) * self.ViewPort_H * (1/2) + self.ViewPort_Y)
    self.point(self.PortX, self.PortY)

#Funcion que cambia el color con que funcionara glClear
 def glClearColor(self, r, g, b):
    self.rc = round(255*r)
    self.gc = round(255*g)
    self.bc = round(255*b)
    self.clearColor = color(self.rc, self.gc, self.bc)

#Funcion para mostrar el archivo
 def archivo(self, filename='out.bmp'):
    self.write(filename)

#Funcion para el color
 def set_color(self, color):
    self.vertexColor =color

 def glColor(self, r, g, b):
    self.rv = round(255*r)
    self.gv = round(255*g)
    self.bv = round(255*b)
    self.vertexColor = color(self.rv, self.gv, self.bv)

#Funcion para el puntos
 def point(self, x, y, color=None):
    self.framebuffer[y][x] = color or self.vertexColor

#Funcion para crear lineas
 def glLine(self, x1, y1, x2, y2):
    #------ y = mx + b ------#
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    #Condicion
    st = dy>dx
    #Condicion de valores 0 en dx
    if dx == 0:
        for y in range(y1, y2+1):
            self.point(x1, y)
    #Condcicion para completar la lineas
    if(st):
        x1,y1 = y1,x1
        x2,y2 = y2,x2
    if(x1>x2):
        x1,x2 = x2,x1
        y1,y2 = y2,y1
    #Valor en x e y
    dx = abs(x2-x1)
    dy = abs(y2-y1)
    llenar = 0
    limite = dx
    y = y1
    #Pendiente
    #m = dy/dx
    for x in range(x1,(x2+1)):
        if(st):
            self.point(y,x)
        else:
            self.point(x,y)
        llenar += dy * 2
        if llenar >= limite:
            y += 1 if y1 < y2 else -1
            limite += 2*dx

#Funcion para dibujar triangulos
 def triangle(self, A, B, C, color=None, texture=None, shader=None, normals = None, texture_coords=(), intensity=1):
    bbox_min, bbox_max = bbox(A, B, C)
    #Llenamos los poligonos con el siguiente Ciclo
    for x in range(bbox_min.x, bbox_max.x + 1):
        for y in range(bbox_min.y, bbox_max.y + 1):
            #Coordenads barycentricas
            w, v, u = baricentricas(A, B, C, V2(x,y))
            #Condicion para evitar numeros negativos
            if (w<0) or (v<0) or (u<0):
                continue
            if color:
                self.vertexColor = color
            #Condicon para los datos de la textura
            if texture:
                tA, tB, tC = texture_coords
                tx = tA.x * w + tB.x * v + tC.x * u
                ty = tA.y * w + tB.y * v + tC.y * u
                #Mandamos los colores
                color = texture.get_color(tx, ty, intensity)
            # si encuentra un shader
            elif shader:
                color = shader(self, bary(w,v,u), vnormals=normals, bcolor=bcolor)
            #Valores para la coordenada z
            z = A.z * w + B.z * v + C.z * u
            #Condicion para evitar numeros negativos
            if (x<0) or (y<0):
                continue
            if x < len(self.zbuffer) and y < len(self.zbuffer[x]) and z > self.zbuffer[x][y]:
                self.point(x, y, color)
                self.zbuffer[x][y] = z


 def loadModelMatrix(self, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0, 0, 0)):
    #Mandamos los datos
    translate = V3(*translate)
    scale = V3(*scale)
    rotate = V3(*rotate)
    #Matriz de translacion
    translatenMatrix = [
        [1, 0, 0, translate.x],
        [0, 1, 0, translate.y],
        [0, 0, 1, translate.z],
        [0, 0, 0,       1    ],
    ]
    #Matriz de escalar
    scaleMatrix = [
        [scale.x, 0, 0, 0],
        [0, scale.y, 0, 0],
        [0, 0, scale.z, 0],
        [0, 0,    0   , 1],
    ]
    #Matrices de rotacion tanto en x, y e z
    #X
    rotateMatriz_X = [
        [1,       0      ,        0      , 0],
        [0, cos(rotate.x), -sin(rotate.x), 0],
        [0, sin(rotate.x),  cos(rotate.x), 0],
        [0,       0      ,        0      , 1]
    ]
    #Y
    rotateMatriz_Y = [
            [ cos(rotate.y), 0,  sin(rotate.y), 0],
            [      0       , 1,        0      , 0],
            [-sin(rotate.y), 0,  cos(rotate.y), 0],
            [      0       , 0,        0      , 1]
    ]
    #Z
    rotateMatriz_Z = [
        [cos(rotate.z), -sin(rotate.z), 0, 0],
        [sin(rotate.z),  cos(rotate.z), 0, 0],
        [      0      ,        0      , 1, 0],
        [      0      ,        0      , 0, 1],
    ]
    #Matriz general de rotacion
    rotateMatriz = multiplicarMatrices(multiplicarMatrices(rotateMatriz_X, rotateMatriz_Y), rotateMatriz_Z)
    #Matriz general que contiene translate, scale, rotate
    self.Model = multiplicarMatrices(multiplicarMatrices(translatenMatrix, rotateMatriz), scaleMatrix)

#Funcion para la viewMatrix
 def loadViewMatrix(self, x, y, z, center):
    M = [
        [x.x, x.y, x.z, 0],
        [y.x, y.y, y.z, 0],
        [z.x, z.y, z.z, 0],
        [ 0 ,  0 ,  0 , 1]
    ]
    O = [
        [1, 0, 0, -center.x],
        [0, 1, 0, -center.y],
        [0, 0, 1, -center.z],
        [0, 0, 0,     1    ]
    ]
    #Multiplicamos las matrices para generar una sola
    self.View = multiplicarMatrices(M, O)

#Funcion que contiene la matriz de projeccion
 def loadProjectionMatrix(self, coeff):
    self.Projection = [
        [1, 0,   0  , 0],
        [0, 1,   0  , 0],
        [0, 0,   1  , 0],
        [0, 0, coeff, 1]
    ]

#Funcion que tiene la matriz de view port
 def loadViewportMatrix(self, x = 0, y = 0):
    self.Viewport = [
        [(self.width*0.5),         0        ,  0  ,  ( x+self.width*0.5) ],
        [       0        , (self.height*0.5),  0  ,  (y+self.height*0.5) ],
        [       0        ,         0        , 128 ,          128         ],
        [       0        ,         0        ,  0  ,          1           ]
    ]

#Funcion para encontrar la transformacion
 def trans(self, vertex, translate=(0,0,0), scale=(1,1,1)):
    return V3(
    round((vertex[0] + translate[0]) * scale[0]),
    round((vertex[1] + translate[1]) * scale[1]),
    round((vertex[2] + translate[2]) * scale[2])
    )

#Funcion que tranforma las MATRICES
 def transform(self, vector):
    nuevoVector = [[vector.x], [vector.y], [vector.z], [1]]
    #Multiplicamos las diferentes matrices resultantes
    modelMultix = multiplicarMatrices(self.Viewport, self.Projection)
    viewMultix = multiplicarMatrices(modelMultix, self.View)
    vpMultix = multiplicarMatrices(viewMultix, self.Model)
    #Vector de transformacion
    vectores = multiplicarMatrices(vpMultix, nuevoVector)
    #transformacion
    transformVector = [
        round(vectores[0][0]/vectores[3][0]),
        round(vectores[1][0]/vectores[3][0]),
        round(vectores[2][0]/vectores[3][0])
    ]
    #print(V3(*transformVector))
    #retornamos los Valores
    return V3(*transformVector)

#Funcion para la vista
 def lookAt(self, eye, center, up):
    z = norm(sub(eye, center))
    x = norm(cross(up, z))
    y = norm(cross(z, x))
    self.loadViewMatrix(x, y, z, center)
    self.loadProjectionMatrix( -1 / length(sub(eye, center)))
    self.loadViewportMatrix()


 def load(self, filename, translate=(0, 0, 0), scale=(1, 1, 1), rotate=(0,0,0), texture=None):
    self.loadModelMatrix(translate, scale, rotate)
    objetos = Obj(filename)
    objetos.read()
    self.light = V3(0,0,1)
    #Ciclo para recorrer las carras
    for face in objetos.faces:
        vcount = len(face)
        #Revisamos cada car
        #Ciclo para recorrer las carras
        for face in objetos.faces:
          vcount = len(face)
          if vcount == 3:
            f1 = face[0][0] - 1
            f2 = face[1][0] - 1
            f3 = face[2][0] - 1

            a = self.transform(objetos.vertices[f1])
            b = self.transform(objetos.vertices[f2])
            c = self.transform(objetos.vertices[f3])

            #Calculamos el vector vnormal
            normal = norm(cross(sub(b,a), sub(c,a)))
            intensity = dot(normal, light)
            if intensity<0:
                continue
            if texture:
                t1 = face[0][1] - 1
                t2 = face[1][1] - 1
                t3 = face[2][1] - 1
                tA = V3(*objetos.texcoords[t1])
                tB = V3(*objetos.texcoords[t2])
                tC = V3(*objetos.texcoords[t3])
                
                #Mandamos los datos a la funcion que se encargara de dibujar el
                self.triangle(a,b,c, texture=texture, texture_coords=(tA,tB,tC), intensity=intensity)
            else:
                grey = round(255*intensity)
                if grey<0:
                    continue
                self.triangle(a,b,c, color=color(grey,grey,grey))
          else:
            # assuming 4
            f1 = face[0][0] - 1
            f2 = face[1][0] - 1
            f3 = face[2][0] - 1
            f4 = face[3][0] - 1   

            vertices = [
                self.transform(objetos.vertices[f1], translate, scale),
                self.transform(objetos.vertices[f2], translate, scale),
                self.transform(objetos.vertices[f3], translate, scale),
                self.transform(objetos.vertices[f4], translate, scale)
            ]

            normal = norm(cross(sub(vertices[0], vertices[1]), sub(vertices[1], vertices[2]))) 
            intensity = dot(normal, light)
            if intensity<0:
                continue

            A, B, C, D = vertices 

            if texture:
                t1 = face[0][1] - 1
                t2 = face[1][1] - 1
                t3 = face[2][1] - 1
                t4 = face[3][1] - 1
                tA = V3(*objetos.texcoords[t1])
                tB = V3(*objetos.texcoords[t2])
                tC = V3(*objetos.texcoords[t3])
                tD = V3(*objetos.texcoords[t4])       

                self.triangle(A, B, C, texture=texture, texture_coords=(tA, tB, tC), intensity=intensity)
                self.triangle(A, C, D, texture=texture, texture_coords=(tA, tC, tD), intensity=intensity)       
            else:
                grey = round(255 * intensity)
                if grey < 0:
                    continue
                self.triangle(A, B, C, color(grey, grey, grey))
                self.triangle(A, C, D, color(grey, grey, grey))  
