import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display

# Variables globales para almacenar estado
img1 = None
img2 = None
points1 = None
points2 = None
triangles = None
detector = None
predictor = None

def load_image(path, size=(256, 256)):
    """Carga y redimensiona una imagen"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    img = cv2.resize(img, size)
    return img

def get_landmarks(img):
    """Detecta puntos de referencia faciales en una imagen"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    if len(rects) == 0:
        raise ValueError("No se detectó ningún rostro.")
    shape = predictor(gray, rects[0])
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)

def add_border_points(points, w, h):
    """Agrega puntos en los bordes de la imagen"""
    return np.vstack([points, [[0,0],[w-1,0],[w-1,h-1],[0,h-1],[w//2,0],[w-1,h//2],[w//2,h-1],[0,h//2]]])

def get_triangles(points, size):
    """Calcula la triangulación de Delaunay para los puntos dados"""
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    
    # Convertir puntos a formato compatible con subdiv2D
    points_list = [(float(p[0]), float(p[1])) for p in points]
    
    # Insertar puntos con manejo de errores
    for p in points_list:
        # Asegurarse de que los puntos estén dentro de los límites
        if p[0] < rect[0] or p[0] >= rect[2] or p[1] < rect[1] or p[1] >= rect[3]:
            continue
        try:
            subdiv.insert(p)
        except Exception:
            continue
    
    # Obtener triángulos
    triangles = subdiv.getTriangleList()
    triangle_indices = []
    
    for t in triangles:
        pts = t.reshape(3, 2)
        idx = []
        for p in pts:
            closest_idx = None
            closest_dist = float('inf')
            for i, ref in enumerate(points):
                dist = np.linalg.norm(p - ref)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i
            if closest_dist < 1.0 and closest_idx is not None:
                idx.append(closest_idx)
        
        if len(idx) == 3:
            triangle_indices.append(idx)
    
    return triangle_indices

def morph_faces(img1, img2, pts1, pts2, alpha, tri_indices):
    """Realiza el morphing entre dos rostros con valor alpha"""
    img1 = np.float32(img1.copy())
    img2 = np.float32(img2.copy())
    h, w = img1.shape[:2]
    morphed = np.zeros(img1.shape, dtype=np.float32)

    # Calcular puntos intermedios
    pts = (1 - alpha) * pts1 + alpha * pts2

    for tri in tri_indices:
        if len(tri) != 3:
            continue

        x, y, z = tri
        
        # Obtener los puntos de los triángulos
        t1 = np.float32([pts1[x], pts1[y], pts1[z]])
        t2 = np.float32([pts2[x], pts2[y], pts2[z]])
        t = np.float32([pts[x], pts[y], pts[z]])

        # Calcular rectángulo que contiene el triángulo
        r = cv2.boundingRect(np.int32(t))
        
        # Ajustar coordenadas relativas al rectángulo
        t1_rect = np.float32(t1 - np.float32([r[0], r[1]]))
        t2_rect = np.float32(t2 - np.float32([r[0], r[1]]))
        t_rect = np.float32(t - np.float32([r[0], r[1]]))

        # Crear máscara para el triángulo actual
        mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16)

        # Recortar las imágenes al rectángulo
        if (r[1]+r[3] <= h and r[0]+r[2] <= w):
            img1_crop = img1[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
            img2_crop = img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
        else:
            continue

        # Calcular matrices de transformación afín (asegurando el formato correcto)
        warp1 = cv2.getAffineTransform(t1_rect, t_rect)
        warp2 = cv2.getAffineTransform(t2_rect, t_rect)

        # Aplicar transformaciones
        warp_img1 = cv2.warpAffine(img1_crop, warp1, (r[2], r[3]))
        warp_img2 = cv2.warpAffine(img2_crop, warp2, (r[2], r[3]))

        # Mezclar las imágenes
        blended = (1 - alpha) * warp_img1 + alpha * warp_img2
        
        # Agregar al resultado final usando la máscara
        morphed[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] += blended * mask

    return np.uint8(np.clip(morphed, 0, 255))

def initialize(face1_path, face2_path, predictor_path="shape_predictor_68_face_landmarks.dat", size=(256, 256)):
    """
    Inicializa el sistema de morphing con dos imágenes de rostros
    
    Args:
        face1_path: Ruta a la primera imagen
        face2_path: Ruta a la segunda imagen
        predictor_path: Ruta al archivo del predictor de landmarks
        size: Tamaño al que se redimensionarán las imágenes
    """
    global img1, img2, points1, points2, triangles, detector, predictor
    
    try:
        # Cargar detector y predictor de landmarks
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        # Cargar imágenes
        img1 = load_image(face1_path, size)
        img2 = load_image(face2_path, size)
        
        print("Imágenes cargadas con éxito.")
        
        # Obtener landmarks
        points1 = get_landmarks(img1)
        points2 = get_landmarks(img2)
        
        print("Landmarks faciales detectados en ambas imágenes.")
        
        # Agregar puntos de borde
        h, w = img1.shape[:2]
        points1 = add_border_points(points1, w, h)
        points2 = add_border_points(points2, w, h)
        
        # Calcular triangulación
        avg_points = (points1 + points2) / 2
        triangles = get_triangles(avg_points, img1.shape)
        
        if len(triangles) == 0:
            print("No se generaron triángulos con los puntos promedio. Intentando con la primera imagen...")
            triangles = get_triangles(points1, img1.shape)
            
            if len(triangles) == 0:
                print("No se generaron triángulos con la primera imagen. Intentando con la segunda imagen...")
                triangles = get_triangles(points2, img1.shape)
        
        print(f"Inicialización completada. Se generaron {len(triangles)} triángulos.")
        print("Usa 'start_interaction()' para iniciar la visualización interactiva.")
        return True
        
    except Exception as e:
        print(f"Error durante la inicialización: {e}")
        import traceback
        traceback.print_exc()
        return False

def update(alpha):
    """Función para actualizar la visualización con el slider interactivo"""
    global img1, img2, points1, points2, triangles
    
    if img1 is None or img2 is None:
        print("Error: No se han inicializado las imágenes. Usa 'initialize()' primero.")
        return
    
    if triangles is None or len(triangles) == 0:
        print("Error: No se generaron triángulos. No se puede realizar el morphing.")
        return
        
    try:
        result = morph_faces(img1, img2, points1, points2, alpha, triangles)
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f"Image Morphing (α={alpha:.2f})")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error durante el morphing: {e}")
        import traceback
        traceback.print_exc()

def start_interaction():
    """Inicia la interacción con el slider"""
    if img1 is None or img2 is None:
        print("Error: No se han inicializado las imágenes. Usa 'initialize()' primero.")
        return
        
    if triangles is None or len(triangles) == 0:
        print("Error: No se generaron triángulos. No se puede realizar el morphing.")
        return
        
    try:
        interact(update, alpha=FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5))
    except Exception as e:
        print(f"Error al iniciar la interacción: {e}")
        import traceback
        traceback.print_exc()

def generate_morph_sequence(output_path="morphing_sequence", num_frames=10):
    """
    Genera una secuencia de imágenes de morphing
    
    Args:
        output_path: Directorio donde guardar las imágenes
        num_frames: Número de frames a generar
    """
    import os
    
    if img1 is None or img2 is None:
        print("Error: No se han inicializado las imágenes. Usa 'initialize()' primero.")
        return
    
    if triangles is None or len(triangles) == 0:
        print("Error: No se generaron triángulos. No se puede realizar el morphing.")
        return
        
    try:
        os.makedirs(output_path, exist_ok=True)
        
        for i in range(num_frames + 1):
            alpha = i / num_frames
            result = morph_faces(img1, img2, points1, points2, alpha, triangles)
            filename = os.path.join(output_path, f"morph_{i:03d}.jpg")
            cv2.imwrite(filename, result)
            print(f"Guardado frame {i}/{num_frames}: {filename}")
    except Exception as e:
        print(f"Error al generar la secuencia: {e}")
        import traceback
        traceback.print_exc()
