def forward_diffusion_with_schedule(image=None, steps=10, schedule_type='linear', 
                            max_noise_level=0.5, size=(128, 128), show=True):
    """
    Aplica el proceso de forward diffusion a una imagen usando una programación beta específica.
    
    Args:
        image (numpy.ndarray o str, optional): Imagen de entrada o ruta a la imagen.
                                              Si es None, carga una imagen por defecto.
        steps (int, optional): Número de pasos de difusión. Por defecto es 10.
        schedule_type (str, optional): Tipo de programación beta ('linear', 'quadratic', 'cosine'). 
                                      Por defecto es 'linear'.
        max_noise_level (float, optional): Nivel máximo de ruido a aplicar. Por defecto es 0.5.
        size (tuple, optional): Tamaño para redimensionar la imagen. Por defecto es (128, 128).
        show (bool, optional): Si es True, muestra las imágenes. Por defecto es True.
        
    Returns:
        tuple: (imagen original, lista de imágenes ruidosas, schedule)
    """
    # Cargar imagen si no se proporciona o si es una ruta
    if image is None:
        original = load_image(None, size=size)
    elif isinstance(image, str):
        original = load_image(image, size=size)
    else:
        # Asumimos que ya es un array de numpy
        original = image
        
    # Generar programación beta
    timesteps = np.linspace(0, 1, steps)
    
    # Seleccionar el tipo de programación
    if schedule_type == 'linear':
        schedule = timesteps * max_noise_level
    elif schedule_type == 'quadratic':
        schedule = (timesteps**2) * max_noise_level
    elif schedule_type == 'cosine':
        schedule = np.cos(timesteps * np.pi / 2) * max_noise_level
    else:
        raise ValueError(f"Tipo de programación '{schedule_type}' no reconocido. "
                        "Opciones válidas: 'linear', 'quadratic', 'cosine'")
    
    # Mostrar imagen original y programación beta si show es True
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Imagen original en el primer subplot
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Imagen original')
        ax1.axis('off')

        # Programación beta en el segundo subplot
        ax2.plot(timesteps, schedule, 'o-', linewidth=2)
        ax2.set_title(f"Programación {schedule_type.capitalize()} β")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Nivel de Ruido")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    # Proceso de forward diffusion
    noisy_images = [original]
    for i in range(1, steps):
        # Usar el valor de la programación para este paso
        noise = np.random.normal(0, schedule[i], original.shape)
        new_image = noisy_images[-1] + noise
        noisy_images.append(np.clip(new_image, 0, 1))
    
    # Mostrar resultados si show es True
    if show:
        fig, axs = plt.subplots(1, steps, figsize=(20, 2))
        for i in range(steps):
            axs[i].imshow(noisy_images[i], cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"t={i}")
        plt.tight_layout()
        plt.show()
    
    return original, noisy_images, schedule

def plot_beta_schedules(num_timesteps=100, show=True):
    """
    Visualiza diferentes programaciones beta (beta schedules) para modelos de difusión.
    
    Las programaciones beta controlan cómo se añade o elimina el ruido en cada paso 
    del proceso de difusión. Diferentes programaciones pueden tener efectos distintos 
    en la calidad y estabilidad del proceso.
    
    Args:
        num_timesteps (int, optional): Número de pasos de tiempo. Por defecto es 100.
        show (bool, optional): Si es True, muestra el gráfico. Por defecto es True.
        
    Returns:
        tuple: (timesteps, linear, quadratic, cosine) - Los arrays con los datos de las programaciones
    """
    # Crear array de timesteps
    timesteps = np.linspace(0, 1, num_timesteps)
    
    # Calcular diferentes programaciones beta
    linear = timesteps
    quadratic = timesteps**2
    cosine = np.cos(timesteps * np.pi / 2)
    
    # Mostrar gráfico si show es True
    if show:
        plt.figure(figsize=(8, 5))
        plt.plot(timesteps, linear, label="Linear β", linewidth=2)
        plt.plot(timesteps, quadratic, label="Quadratic β", linewidth=2)
        plt.plot(timesteps, cosine, label="Cosine β", linewidth=2)
        plt.title("Beta Schedules para Modelos de Difusión", fontsize=14)
        plt.xlabel("Timesteps", fontsize=12)
        plt.ylabel("Cantidad de Ruido", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return timesteps, linear, quadratic, cosine
"""
Módulo de difusión para procesos de forward diffusion y denoising en imágenes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imread
import os


def load_image(image_path=None, size=(128, 128)):
    """
    Carga una imagen desde un archivo o usa una imagen por defecto.
    
    Args:
        image_path (str, optional): Ruta a la imagen. Si es None, usa una imagen por defecto.
        size (tuple, optional): Tamaño para redimensionar la imagen. Por defecto es (128, 128).
        
    Returns:
        numpy.ndarray: Imagen en escala de grises normalizada.
    """
    if image_path is None:
        # Usar imagen por defecto
        original = rgb2gray(data.astronaut())
    else:
        # Cargar imagen desde archivo
        try:
            # Verificar si el archivo existe
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"No se encontró el archivo: {image_path}")
                
            # Cargar imagen utilizando la función imread de skimage
            img = imread(image_path)
            
            # Asegurar que la imagen sea array numérico
            if not isinstance(img, np.ndarray):
                raise ValueError("La imagen cargada no es un array numérico")
                
            # Convertir a escala de grises si es RGB
            if len(img.shape) == 3:
                original = rgb2gray(img)
            else:
                # Convertir a float para asegurar compatibilidad
                original = img.astype(float)
                
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            print("Usando imagen por defecto...")
            original = rgb2gray(data.astronaut())
    
    # Redimensionar
    original = resize(original, size)
    
    # Normalizar la imagen (asegurarse de que esté en el rango [0,1])
    original_min = original.min()
    original_max = original.max()
    
    # Evitar división por cero
    if original_max > original_min:
        original = (original - original_min) / (original_max - original_min)
    else:
        # Si todos los píxeles tienen el mismo valor
        original = np.zeros_like(original)
    
    return original


def forward_diffusion(image=None, steps=10, noise_level=0.1, size=(128, 128), show=True):
    """
    Aplica el proceso de forward diffusion a una imagen.
    
    Args:
        image (numpy.ndarray o str, optional): Imagen de entrada o ruta a la imagen. 
                                                Si es None, carga una imagen por defecto.
        steps (int, optional): Número de pasos de difusión. Por defecto es 10.
        noise_level (float, optional): Nivel de ruido a añadir en cada paso. Por defecto es 0.1.
        size (tuple, optional): Tamaño para redimensionar la imagen. Por defecto es (128, 128).
        show (bool, optional): Si es True, muestra las imágenes. Por defecto es True.
        
    Returns:
        tuple: (imagen original, lista de imágenes ruidosas)
    """
    # Cargar imagen si no se proporciona o si es una ruta
    if image is None:
        original = load_image(None, size=size)
    elif isinstance(image, str):
        original = load_image(image, size=size)
    else:
        # Asumimos que ya es un array de numpy
        original = image
        
    # Mostrar imagen original si show es True
    if show:
        plt.figure(figsize=(5, 5))
        plt.imshow(original, cmap='gray')
        plt.title('Imagen original')
        plt.axis('off')
        plt.show()
    
    # Proceso de forward diffusion
    noisy_images = [original]
    for i in range(1, steps):
        noise = np.random.normal(0, noise_level, original.shape)
        new_image = noisy_images[-1] + noise
        noisy_images.append(np.clip(new_image, 0, 1))
    
    # Mostrar resultados si show es True
    if show:
        fig, axs = plt.subplots(1, steps, figsize=(20, 2))
        for i in range(steps):
            axs[i].imshow(noisy_images[i], cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"t={i}")
        plt.tight_layout()
        plt.show()
    
    return original, noisy_images


def denoising(image=None, steps=10, initial_noise=1.0, sigma_factor=1.5, 
              original_weight_factor=0.1, size=(128, 128), show=True):
    """
    Simula el proceso de denoising en una imagen.
    
    Args:
        image (numpy.ndarray o str, optional): Imagen de entrada o ruta a la imagen. 
                                                Si es None, carga una imagen por defecto.
        steps (int, optional): Número de pasos de denoising. Por defecto es 10.
        initial_noise (float, optional): Nivel de ruido inicial. Por defecto es 1.0.
        sigma_factor (float, optional): Factor para ajustar sigma en el filtro gaussiano. Por defecto es 1.5.
        original_weight_factor (float, optional): Factor de peso para la imagen original. Por defecto es 0.1.
        size (tuple, optional): Tamaño para redimensionar la imagen. Por defecto es (128, 128).
        show (bool, optional): Si es True, muestra las imágenes. Por defecto es True.
        
    Returns:
        tuple: (imagen original, imagen con ruido, lista de imágenes denoised)
    """
    # Cargar imagen si no se proporciona o si es una ruta
    if image is None:
        original = load_image(None, size=size)
    elif isinstance(image, str):
        original = load_image(image, size=size)
    else:
        # Asumimos que ya es un array de numpy
        original = image
    
    # Crear imagen ruidosa
    noisy = original + np.random.normal(0, initial_noise, original.shape)
    noisy = np.clip(noisy, 0, 1)
    
    # Mostrar imagen ruidosa si show es True
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original, cmap='gray')
        ax1.set_title('Original')
        ax1.axis('off')
        ax2.imshow(noisy, cmap='gray')
        ax2.set_title('Ruidosa')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Simular proceso de denoising
    images = [noisy]
    for i in range(1, steps):
        sigma = sigma_factor * (steps - i) / steps
        img = gaussian_filter(images[-1], sigma=sigma)
        img = img * (1 - original_weight_factor*i) + original * original_weight_factor*i
        images.append(np.clip(img, 0, 1))
    
    # Mostrar resultados si show es True
    if show:
        fig, axs = plt.subplots(1, steps, figsize=(20, 2))
        for i in range(steps):
            axs[i].imshow(images[i], cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f"Step {i}")
        plt.tight_layout()
        plt.show()
    
    return original, noisy, images


def diffusion_process(image_path=None, forward_steps=10, denoising_steps=10, 
                     noise_level=0.1, initial_noise=1.0, size=(128, 128)):
    """
    Ejecuta el proceso completo de difusión: forward diffusion seguido de denoising.
    
    Args:
        image_path (str o numpy.ndarray, optional): Ruta a la imagen o array de numpy. 
                                                    Si es None, usa una imagen por defecto.
        forward_steps (int, optional): Número de pasos de forward diffusion. Por defecto es 10.
        denoising_steps (int, optional): Número de pasos de denoising. Por defecto es 10.
        noise_level (float, optional): Nivel de ruido para forward diffusion. Por defecto es 0.1.
        initial_noise (float, optional): Nivel de ruido inicial para denoising. Por defecto es 1.0.
        size (tuple, optional): Tamaño para redimensionar la imagen. Por defecto es (128, 128).
        
    Returns:
        tuple: (imagen original, imágenes ruidosas, imágenes denoised)
    """
    # Cargar imagen
    if image_path is None:
        original = load_image(None, size)
    elif isinstance(image_path, str):
        original = load_image(image_path, size)
    else:
        # Asumimos que ya es un array de numpy
        original = image_path
    
    # Mostrar imagen original
    plt.figure(figsize=(5, 5))
    plt.imshow(original, cmap='gray')
    plt.title('Imagen original')
    plt.axis('off')
    plt.show()
    
    # Forward diffusion
    print("Proceso de Forward Diffusion:")
    _, noisy_images = forward_diffusion(original, steps=forward_steps, 
                                       noise_level=noise_level, show=True)
    
    # Denoising
    print("\nProceso de Denoising:")
    _, _, denoised_images = denoising(original, steps=denoising_steps, 
                                     initial_noise=initial_noise, show=True)
    
    return original, noisy_images, denoised_images