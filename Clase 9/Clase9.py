import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from pytorch_pretrained_biggan import (one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal, one_hot_from_int)
import ast
use_gpu = True if torch.cuda.is_available() else False

def plot_progan(model, num_images=12):
    noise, _ = model.buildNoiseData(num_images)
    with torch.no_grad():
        generated_images = model.test(noise)
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), nrow=6, scale_each=True, normalize=True)
    fig = plt.figure(figsize=(18, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Generated examples from a ProGAN")
    plt.axis('off')
    plt.show()

def plot_stylegan(model, device, num_images=12):
    latents = torch.randn(num_images, 512, device=device)
    with torch.no_grad():
        imgs = model(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0 # normalization to 0..1 range
    imgs = imgs.cpu()
    imgs = torchvision.utils.make_grid(imgs, nrow=6)
    fig = plt.figure(figsize=(18, 6))
    plt.imshow(imgs.permute(1, 2, 0).detach().numpy())
    plt.title("Generated examples from a StyleGAN")
    plt.axis('off')
    plt.show()
    
def plot_interpolation(model, device, nb_interp = 10, n_max_col=11, title=None):
    fixed_latents = [torch.randn(1, 512, device=device) for _ in range(nb_interp)]
    latents = []
    for i in range(len(fixed_latents)-1):
        latents.append(fixed_latents[i] + (fixed_latents[i + 1] - fixed_latents[i]) 
                       * torch.arange(0, 1, 0.1, device=device).unsqueeze(1))
    latents.append(fixed_latents[-1])
    latents = torch.cat(latents, dim=0)
    
    num_images = len(latents)
    num_columns = min(num_images, n_max_col)
    num_rows = (num_images - 1) // num_columns + 1

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 12))
    
    with torch.no_grad():
        for idx, latent in enumerate(latents):
            latent = latent.to(device)
            img = model(latent.unsqueeze(0))
            img = img.clamp_(-1, 1).add_(1).div_(2.0)        
            img = img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()           
            row = idx // num_columns
            col = idx % num_columns

            ax = axes[row, col] if num_rows > 1 else axes[col]

            ax.imshow(img)
            ax.axis('off')

    # Ocultar ejes que no contienen imágenes
    for i in range(num_images, num_rows * num_columns):
        row = i // num_columns
        col = i % num_columns

        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')
        ax.set_visible(False)

    if title is not None:
        fig.text(0.5, 0.75, title, ha='center', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.7)
    plt.show()
    
def plot_images(
    model, category, num_samples = 12, truncation = 0.4, n_max_col=12
):
    
    try:
        class_vector = one_hot_from_names(category, batch_size=num_samples)
    except:
        return "Class not found."
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=num_samples)
    
    noise_vector = torch.from_numpy(noise_vector)
    try:
        class_vector = torch.from_numpy(class_vector)
    except:
        return "Class not found."
    
    with torch.no_grad():
        images = model(noise_vector, class_vector, truncation)
    
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0
    
    num_images = len(images)
    num_columns = min(num_images, n_max_col)
    num_rows = (num_images - 1) // num_columns + 1
    
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 5))
   
    for idx, img in enumerate(images):
        
        img = img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()           
        row = idx // num_columns
        col = idx % num_columns

        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.imshow(img)
        ax.axis('off')

    # Ocultar ejes que no contienen imágenes
    for i in range(num_images, num_rows * num_columns):
        row = i // num_columns
        col = i % num_columns

        ax = axes[row, col] if num_rows > 1 else axes[col]
        ax.axis('off')
        ax.set_visible(False)
    plt.show()

def load_data(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return ast.literal_eval(content)

def transform_dict(data_dict):
    transformed_dict = {}
    for key, value in data_dict.items():
        names = [name.strip() for name in value.split(',')]
        transformed_dict[key] = names
    return transformed_dict

def find_id_by_name(name, data_dict):
    for key, value in data_dict.items():
        if name.lower() in [string.lower() for string in value]:
            return key
    return None  # If the name is not found

def latent_walk(model, data_dictionary, class_1, class_2, truncation = 0.5, num_cols = 10):
    class_vector = np.zeros((num_cols,1000),dtype=np.float32)
    noise_vector = np.zeros((num_cols,128),dtype=np.float32)
    id_1 = find_id_by_name(class_1,data_dictionary)
    id_2 = find_id_by_name(class_2,data_dictionary)
    if id_1 is None:
        return "{} not found.".format(class_1)
    if id_2 is None:
        return "{} not found.".format(class_2)
    class_vector[0,id_1]=1 # START CLASS
    class_vector[num_cols-1,id_2]=1 # END CLASS   
    step = class_vector[num_cols-1,] - class_vector[0,]
    for k in range(1,num_cols-1):
        class_vector[k,] = class_vector[0,] + (k/(num_cols-1))*step
    noise_vector[0,:] = truncated_noise_sample(truncation=truncation, batch_size=1)
    noise_vector[num_cols-1,:] = truncated_noise_sample(truncation=truncation, batch_size=1)
    step = noise_vector[num_cols-1,]-noise_vector[0,]
    for k in range(1,num_cols-1):
        noise_vector[k,] = noise_vector[0,] + (k/(num_cols-1))*step
    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)
    # Generate an image
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)
    output = output.numpy().transpose(0, 2, 3, 1)
    if output.max() > 1.0:
        output = output / 255.0
    elif output.min() < 0.0:
        output = (output + 1.0) / 2.0
    plt.figure(figsize=(int(2*num_cols),2))
    plt.subplots_adjust(hspace=0,wspace=0)
    for k in range(num_cols):
        plt.subplot(1,num_cols,k+1)        
        plt.axis('off')
        plt.imshow(output[k])
    plt.suptitle("From {} to {}".format(class_1,class_2))
    plt.show()

def plot_style_mixing(model, device, mixing_cutoff=5, num_images=5):
    """
    Visualiza style mixing usando dos vectores z diferentes.
    
    Parameters:
        model: StyleGAN model con g_mapping y g_synthesis
        device: CPU o CUDA
        mixing_cutoff: índice de capa donde se mezcla el estilo (por ejemplo 5)
        num_images: número de imágenes de mezcla a generar
    """
    z1 = torch.randn(1, 512, device=device)
    z2 = torch.randn(1, 512, device=device)

    # Estilo latente
    w1 = model.g_mapping(z1)
    w2 = model.g_mapping(z2)

    # Duplicamos los estilos para todas las capas (18 en StyleGAN)
    w1 = w1.repeat(1, 18, 1)
    w2 = w2.repeat(1, 18, 1)

    images = []

    with torch.no_grad():
        for cutoff in range(mixing_cutoff, mixing_cutoff + num_images):
            # Mezclamos: w1 para las primeras 'cutoff' capas, w2 para el resto
            w_mixed = w1.clone()
            w_mixed[:, cutoff:] = w2[:, cutoff:]

            # Generar imagen
            img = model.g_synthesis(w_mixed)
            img = (img.clamp(-1, 1) + 1) / 2.0  # Normalizar
            images.append(img)

    # Visualización
    images = torch.cat(images, dim=0).cpu()
    grid = torchvision.utils.make_grid(images, nrow=num_images, padding=2)
    plt.figure(figsize=(num_images * 3, 3))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(f"Style Mixing (cutoff desde capa {mixing_cutoff})")
    plt.axis('off')
    plt.show()

def plot_style_mixing_with_refs(model, device, cutoff=6, z_dim=512, seed=None):
    model.eval()
    if seed:
        torch.manual_seed(seed)

    # Generar dos vectores latentes z1 y z2
    z1 = torch.randn(1, z_dim, device=device)
    z2 = torch.randn(1, z_dim, device=device)

    # Separar el modelo StyleGAN
    g_mapping = model.g_mapping
    g_synthesis = model.g_synthesis

    # Mapear a espacio intermedio w
    with torch.no_grad():
        w1 = g_mapping(z1)  # shape: [1, 18, 512]
        w2 = g_mapping(z2)  # shape: [1, 18, 512]

        # Mezclar estilos según el cutoff
        w_mix = w1.clone()
        w_mix[:, cutoff:] = w2[:, cutoff:]

        # Generar imágenes
        img1 = g_synthesis(w1)
        img_mix = g_synthesis(w_mix)
        img2 = g_synthesis(w2)

    # Preprocesar imágenes para visualización
    def postprocess(img):
        img = (img.clamp(-1, 1) + 1) / 2.0
        return img.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()

    img1 = postprocess(img1)
    img_mix = postprocess(img_mix)
    img2 = postprocess(img2)

    # Mostrar imágenes
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1)
    axes[0].set_title("w1 (early style)")
    axes[0].axis("off")

    axes[1].imshow(img_mix)
    axes[1].set_title(f"Mixed @ cutoff={cutoff}")
    axes[1].axis("off")

    axes[2].imshow(img2)
    axes[2].set_title("w2 (late style)")
    axes[2].axis("off")

    plt.suptitle("Style Mixing in StyleGAN", fontsize=16)
    plt.tight_layout()
    plt.show()
    
def plot_truncation_effect_biggan(model, class_id=207, z_dim=128, truncation_values=[1.0, 0.7, 0.5, 0.3, 0.1], num_cols=6):
    model.eval()
    
    num_rows = len(truncation_values)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5))
    fig.suptitle("Efecto del Truncation Trick en BigGAN", fontsize=18)
    
    for row_idx, trunc_val in enumerate(truncation_values):
        for col_idx in range(num_cols):
            # Sample z and one-hot y
            z = truncated_noise_sample(truncation=trunc_val, batch_size=1, dim_z=z_dim)
            z = torch.from_numpy(z)
            y = one_hot_from_int([class_id], batch_size=1)
            y = torch.from_numpy(y)
            # Forward pass
            with torch.no_grad():
                output = model(z, y, truncation=trunc_val)
                img = output.squeeze(0).cpu()
                img = (img + 1) / 2.0  # Rescale to [0, 1]
            # Plot
            ax = axes[row_idx, col_idx]
            npimg = np.transpose(img.numpy(), (1, 2, 0))
            ax.imshow(npimg)
            ax.axis('off')
            
            # Añadir la etiqueta del valor de truncation solo a la primera columna
            if col_idx == 0:
                # Reactivar el eje y para mostrar la etiqueta
                ax.axis('on')
                ax.tick_params(left=False, bottom=False, labelleft=True, labelbottom=False)
                # Eliminar las marcas del eje pero mantener la etiqueta
                ax.set_xticks([])
                ax.set_yticks([])
                # Establecer la etiqueta del eje y con el valor de truncation
                ax.set_ylabel(f"trunc={trunc_val:.1f}", fontsize=12, rotation=90, labelpad=10, 
                              verticalalignment='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()