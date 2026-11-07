#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import torch
import torchvision

from lucent.modelzoo import *
from lucent.misc.io import show
import lucent.optvis.objectives as objectives
import lucent.optvis.param as param
import lucent.optvis.render as render
import lucent.optvis.transform as transform
from lucent.misc.channel_reducer import ChannelReducer
import matplotlib.pyplot as plt

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


@torch.no_grad()
def get_layer(model, layer, X):
    hook = render.ModuleHook(getattr(model, layer))
    model(X)
    hook.close()
    return hook.features


@objectives.wrap_objective()
def dot_compare(layer, acts, batch=1):
    def inner(T):
        pred = T(layer)[batch]
        return -(pred * acts).sum(dim=0, keepdims=True).mean()

    return inner

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image
# In[ ]:
def render_vis(
    model,
    objective_f,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    preprocess=True,
    fixed_image_size=None,
):
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    # Upsample images smaller than 224
    image_shape = image_f().shape
    if fixed_image_size is not None:
        new_size = fixed_image_size
    elif image_shape[2] < 224 or image_shape[3] < 224:
        new_size = 224
    else:
        new_size = None
    if new_size:
        transforms.append(
            torch.nn.Upsample(size=new_size, mode="bilinear", align_corners=True)
        )

    transform_f = transform.compose(transforms)

    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    images = []
    try:
        for i in (range(1, max(thresholds) + 1)):
            def closure():
                optimizer.zero_grad()
                try:
                    model(transform_f(image_f()))
                except:
                    pass
                loss = objective_f(hook)
                loss.backward()
                return loss
                
            optimizer.step(closure)
            if i in thresholds:
                image = tensor_to_img_array(image_f())
                images.append(image)
    except KeyboardInterrupt:
        print("Interrupted optimization at step {:d}.".format(i))
        images.append(tensor_to_img_array(image_f()))
    return images

def activations(
    img,
    model,
    layer,
):
    # First we need, to normalize and resize the image
    img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
    normalize = (
        transform.preprocess_inceptionv1()
        if model._get_name() == "InceptionV1"
        else transform.normalize()
    )
    transforms = transform.standard_transforms.copy() + [
        normalize,
        torch.nn.Upsample(size=224, mode="bilinear", align_corners=True),
    ]
    transforms_f = transform.compose(transforms)
    # shape: (1, 3, original height of img, original width of img)
    img = img.unsqueeze(0)
    # shape: (1, 3, 224, 224)
    img = transforms_f(img)

    # Here we compute the activations of the layer `layer` using `img` as input
    # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
    acts = get_layer(model, layer, img)[0]
    # shape: (layer_height, layer_width, layer_channels)
    acts = acts.permute(1, 2, 0)
    # shape: (layer_height*layer_width, layer_channels)
    acts = acts.view(-1, acts.shape[-1])
    acts_np = acts.cpu().numpy()
    return acts_np

def render_activation_grid(
    img,
    model,
    layer,
    cell_image_size=60,
    n_groups=6,
    n_steps=1024,
    batch_size=64,
    pad=0,
    show_image=False,
):
    # First wee need, to normalize and resize the image
    img = torch.tensor(np.transpose(img, [2, 0, 1])).to(device)
    normalize = (
        transform.preprocess_inceptionv1()
        if model._get_name() == "InceptionV1"
        else transform.normalize()
    )
    transforms = transform.standard_transforms.copy() + [
        normalize,
        torch.nn.Upsample(size=224, mode="bilinear", align_corners=True),
    ]
    transforms_f = transform.compose(transforms)
    # shape: (1, 3, original height of img, original width of img)
    img = img.unsqueeze(0)
    # shape: (1, 3, 224, 224)
    img = transforms_f(img)

    # Here we compute the activations of the layer `layer` using `img` as input
    # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
    acts = get_layer(model, layer, img)[0]
    # shape: (layer_height, layer_width, layer_channels)
    acts = acts.permute(1, 2, 0)
    # shape: (layer_height*layer_width, layer_channels)
    acts = acts.view(-1, acts.shape[-1])
    acts_np = acts.cpu().numpy()
    nb_cells = acts.shape[0]

    # negative matrix factorization `NMF` is used to reduce the number
    # of channels to n_groups. This will be used as the following.
    # Each cell image in the grid is decomposed into a sum of
    # (n_groups+1) images. First, each cell has its own set of parameters
    #  this is what is called `cells_params` (see below). At the same time, we have
    # a of group of images of size 'n_groups', which also have their own image parametrized
    # by `groups_params`. The resulting image for a given cell in the grid
    # is the sum of its own image (parametrized by `cells_params`)
    # plus a weighted sum of the images of the group. Each each image from the group
    # is weighted by `groups[cell_index, group_idx]`. Basically, this is a way of having
    # the possibility to make cells with similar activations have a similar image, because
    # cells with similar activations will have a similar weighting for the elements
    # of the group.
    if n_groups > 0:
        reducer = ChannelReducer(n_groups, "NMF")
        groups = reducer.fit_transform(acts_np)
        groups /= groups.max(0)
    else:
        groups = np.zeros([])
    # shape: (layer_height*layer_width, n_groups)
    groups = torch.from_numpy(groups)

    # Parametrization of the images of the groups (we have 'n_groups' groups)
    groups_params, groups_image_f = param.fft_image(
        [n_groups, 3, cell_image_size, cell_image_size]
    )
    # Parametrization of the images of each cell in the grid (we have 'layer_height*layer_width' cells)
    cells_params, cells_image_f = param.fft_image(
        [nb_cells, 3, cell_image_size, cell_image_size]
    )

    # First, we need to construct the images of the grid
    # from the parameterizations

    def image_f():
        groups_images = groups_image_f()
        cells_images = cells_image_f()
        X = []
        for i in range(nb_cells):
            x = 0.7 * cells_images[i] + 0.5 * sum(
                groups[i, j] * groups_images[j] for j in range(n_groups)
            )
            X.append(x)
        X = torch.stack(X)
        return X

    # make sure the images are between 0 and 1
    image_f = param.to_valid_rgb(image_f, decorrelate=True)

    # After constructing the cells images, we sample randomly a mini-batch of cells
    # from the grid. This is to prevent memory overflow, especially if the grid
    # is large.
    def sample(image_f, batch_size):
        def f():
            X = image_f()
            inds = torch.randint(0, len(X), size=(batch_size,))
            inputs = X[inds]
            # HACK to store indices of the mini-batch, because we need them
            # in objective func. Might be better ways to do that
            sample.inds = inds
            return inputs

        return f

    image_f_sampled = sample(image_f, batch_size=batch_size)

    # Now, we define the objective function

    def objective_func(model):
        # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
        pred = model(layer)
        # use the sampled indices from `sample` to get the corresponding targets
        target = acts[sample.inds].to(pred.device)
        # shape: (batch_size, layer_channels, 1, 1)
        target = target.view(target.shape[0], target.shape[1], 1, 1)
        dot = (pred * target).sum(dim=1).mean()
        return -dot

    obj = objectives.Objective(objective_func)

    def param_f():
        # We optimize the parametrizations of both the groups and the cells
        params = list(groups_params) + list(cells_params)
        return params, image_f_sampled

    results = render_vis(
        model,
        obj,
        param_f,
        thresholds=(n_steps,),
        fixed_image_size=cell_image_size,
    )
    # shape: (layer_height*layer_width, 3, grid_image_size, grid_image_size)
    imgs = image_f()
    imgs = imgs.cpu().data
    imgs = imgs[:, :, 2:-2, 2:-2]
    # turn imgs into a grid
    grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(nb_cells)), padding=pad)
    grid = grid.permute(1, 2, 0)
    grid = grid.numpy()
    if show_image:
        render.show(grid)
    return grid

from collections import OrderedDict

class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook

def semantic_dictionary(imagen, indices, model, layer, K=3):
    act = activations(imagen, model, layer)
    # Redimensionar la imagen a 224x224 si no lo está
    if imagen.shape[:2] != (224, 224):
        imagen_redimensionada = np.resize(imagen.copy(), (224, 224, imagen.shape[2]))
    else:
        imagen_redimensionada = imagen.copy()
    
    i, j = indices
    if i < 0 or j < 0 or i * 16 + 16 > 224 or j * 16 + 16 > 224:
        raise ValueError("Los índices están fuera de rango para obtener una parte" 
                         "de la imagen de 16x16 en esa posición")
    parte_de_imagen = imagen_redimensionada[i * 16:i * 16 + 16, j * 16:j * 16 + 16, :]
    
    # Color del borde
    color_nuevo = [255, 0, 0]

    # Iterar sobre los píxeles de la sección de la imagen y cambiar su color
    # Cambiar el color del borde de 1 píxel
    for fila in range(parte_de_imagen.shape[0]):
        for columna in range(parte_de_imagen.shape[1]):
            # Si estamos en el borde de la sección
            if (fila == 0 or fila == parte_de_imagen.shape[0] - 1 or 
                columna == 0 or columna == parte_de_imagen.shape[1] - 1):
                # Asignar el nuevo color al píxel del borde
                parte_de_imagen[fila, columna] = color_nuevo

    # Cambia el color de los píxeles en 'imagen_redimensionada' utilizando 'parte_de_imagen'
    imagen_redimensionada[i * 16:i * 16 + 16, j * 16:j * 16 + 16, :] = parte_de_imagen
    
    # Normalizar para asegurar que los valores estén en el rango [0, 1]
    imagen_redimensionada = imagen_redimensionada.astype(float) / 255.0
    
    mi_array = act[i * 14 + j][:12]
    # Convertir el array a strings con dos decimales
    elementos_formateados = ["{:.2f}".format(valor) if valor != 0 else "0" for valor in mi_array]

    # Crear la cadena final eliminando elementos vacíos y separando por comas
    mi_array_str = "[" + ",".join(elemento for elemento in elementos_formateados if elemento) + ",...]"
    print(f"a{i},{j} =",mi_array_str)
    
    indices_ordenados_ascendente = np.argsort(act[i * 14 + j]) # Top-K
    valores_ordenados_ascendente = np.sort(act[i * 14 + j])
    
    # Obtener los índices de los 4 elementos más grandes en orden descendente
    indices_maximos_descendente = indices_ordenados_ascendente[-K:][::-1]
    valores_maximos_descendente = valores_ordenados_ascendente[-K:][::-1]
    
    indices_maximos_descendente = list(map(str, indices_maximos_descendente))
    valores_maximos_descendente = list(map(lambda x: str(round(x)), valores_maximos_descendente))

    names = ['Channel ' + idx1 + ' (' + idx2 + ')' 
             for idx1, idx2 in zip(indices_maximos_descendente, valores_maximos_descendente)]
    names = ["Activation Vector"]+names
    
    list_of_images = []
    list_of_images.append(imagen_redimensionada.tolist())
    for n in indices_maximos_descendente:
        objective_f = layer + ":" + str(n)
        list_of_images.append(render.render_vis(model, 
                                                objective_f, 
                                                show_inline=False, 
                                                show_image=False, 
                                                progress=False))
        
    show(list_of_images, labels=names)

def plot_grid_layers(list_of_images, names):
    num_images = len(list_of_images)
    num_cols = len(list_of_images)  # Number of columns in the subplot grid
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows based on the number of images

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))

    # Flatten the axes if necessary
    axes = axes.flatten()

    # Loop through the images and plot them in subplots
    for i, image in enumerate(list_of_images):
        axes[i].imshow(image)
        axes[i].axis('off')  # Turn off axis labels

        # Add a title if needed
        axes[i].set_title(f'{names[i]}')

    # Hide any extra empty subplots, if present
    for j in range(num_images, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()