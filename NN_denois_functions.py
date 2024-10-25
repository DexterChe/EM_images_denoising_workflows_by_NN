"""
Created on Wed May 17
Author: Dmitry Chezganov

The file contains the function to use for NN image denosing

"""
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askdirectory
from tkinter.messagebox import askyesno
from tk_r_em import load_network
from matplotlib_scalebar.scalebar import ScaleBar
import tifffile
import hyperspy.api as hs
import sys

# create folder if does not exist
def make_folder(path_to_folder_to_read, folder_name):
    if not os.path.exists(os.path.join((path_to_folder_to_read + folder_name))):
        try:
            os.makedirs(os.path.join(path_to_folder_to_read + folder_name))
        except OSError: print(f'Creation of the directory {os.path.join(path_to_folder_to_read + folder_name)} failed. Probably already present.')
        else:
            print(f'Successfully created the directory {os.path.join(path_to_folder_to_read + folder_name)}')
    path_to_save = path_to_folder_to_read +folder_name+'/'
    return path_to_save

# fuction to adjust data range
def adjust_data_range(image: np.ndarray,
                        p_low: float=0.5,
                        p_high: float=99.8) -> np.ndarray:
    """Clip image values between the p_low and p_high percentile values
    Args:
        image: 2D numpy array
        p_low: lower percentile value (default: 5)
        p_high: higher percentile value (default: 95)
    Return:
        clipped image
    """
    if image is None:
        raise ValueError("Image is None")
    if image.ndim != 2:
        raise ValueError("Image must be 2D")
    low, top = np.percentile(image, (p_low, p_high))
    image = np.clip(image, low, top) # clip the image between low and top
    return image

# def adjust_data_range(image,
#                         p_low=0.5,
#                         p_high=99.8):
#     if image is None:
#         raise ValueError("Image is None")
#     if image.ndim != 2:
#         raise ValueError("Image must be 2D")
#     low, top = np.percentile(image, (p_low, p_high))
#     image = np.clip(image, low, top) # clip the image between low and top
#     return image

def convert_float32_to_uint8(float32_array):
    """
    This code converts a float64 array to a uint8 array.

    Args:
        float64_array (np.ndarray): A 2D array of float64 values.

    Returns:
        np.ndarray: A 2D array of uint8 values.
    """
    # Check that the input is of the correct type and shape
    # ? Changed here - check
    # if float64_array.dtype != np.float64:
    #     raise TypeError('float64_array must be a numpy array of type float64')
    
    # if len(float64_array.shape) != 2:
    #     raise ValueError('float64_array must be a 2D numpy array')
    
    # Find the range of values in the float64 array
    min_value = np.min(float32_array)
    max_value = np.max(float32_array)
    
    # Scale the float64 array to the range of 0-255
    scaled_array = (float32_array - min_value) / (max_value - min_value) * 255.0
    
    # Convert the scaled array to uint8
    uint8_array = scaled_array.astype(np.uint8)
    
    return uint8_array

# Save denoised image as pure TIFF image (32-bit) without scalebar but with metadata in
def save_tif_32bit_image_denoised(image, file_name, outputpath, pixelsize, unit, model_name):
    """Save a 32-bit image to a tif file.
    
    Args:
        image: Image to save.
        file_name: Name of the image file.
        outputpath: Path to the output directory.
        pixelsize: Pixel size in µm.
        unit: Unit of pixel size.
        model_name: Name of the model used to denoise the image.
    """
    if image.dtype != 'float32':
        image = image.astype('float32')
    tifffile.imwrite(outputpath + '/' + file_name + f'-denoise-{model_name}_32bit.tif', 
                        image, 
                        imagej=True, 
                        resolution=(1./pixelsize, 1./pixelsize),
                        metadata={'unit': unit},
                        )

def save_tif_32bit_image_original(image, file_name, outputpath, pixelsize, unit, model_name):
    """Save a 32-bit original image to a tif file.
    
    Args:
        image: Image to save.
        file_name: Name of the image file.
        outputpath: Path to the output directory.
        pixelsize: Pixel size in µm.
        unit: Unit of pixel size.
        model_name: Name of the model used to denoise the image.
    """
    if image.dtype != 'float32':
        image = image.astype('float32')
    tifffile.imwrite(outputpath + '/' + file_name + f'-original_32bit.tif', 
                        image, 
                        imagej=True, 
                        resolution=(1./pixelsize, 1./pixelsize),
                        metadata={'unit': unit})

# def plot_original_vs_denoised(original_image, 
#                                 denoised_image, 
#                                 net_name,
#                                 file_name, 
#                                 cmap,
#                                 save_path,
#                                 show=True, 
#                                 scale=1,
#                                 units='nm',
#                                 px_size=1,
#                                 HT='add HT here',
#                                 mag='add mag here',

#                                 ):
#     # add scale
#     image_scale_x = image_scale_y = scale

#     image_FOV_x = original_image.data.shape[1] * image_scale_x
#     image_FOV_y = original_image.data.shape[0] * image_scale_y
#     image_extent = [0, image_FOV_x, 0, image_FOV_y]
#     px_size = image_scale_x

#     #change units in case of diffraction to support the scalebar
#     original_units = units
#     if units == '1/nm':
#         units = 'nm'

#     scalebar = ScaleBar(
#                 dx=px_size, 
#                 units=units,
#                 location='lower right',
#                 frameon=True,
#                 scale_loc='top',
#                 length_fraction='0.15',
#                 height_fraction='0.005'
#                 )
#     # Add magnification and pixel size to plot using .bcf file
#     font_dict = {'size': 6,
#                 'color': 'white',
#                 'weight': 'bold',
#                 'verticalalignment': 'bottom'
#                 }

#     fig, axs = plt.subplots(1, 2, 
#                             figsize=(12,6.5),
#                             # figsize=(7.5, 3.5), 
#                             sharex=True, 
#                             sharey=True)
#     # cmap = 'gray' # 'turbo'
#     # cmap = 'hot'
#     if original_image.ndim != 2:
#         picture = original_image[0,:,:,0]
#     else:
#         picture = original_image
#     axs[0].imshow(picture, 
#                     # extent=image_extent, 
#                     cmap=cmap)
#     axs[0].set_title('Original')
#     #! set data zone text
#     axs[0].text(0.02, 0.02, f"File: {file_name.split(' ')[0]} px: {round(px_size, 5)}\nHV: {int(HT)} kV  Mag: {mag}kx", transform=axs[0].transAxes, fontdict=font_dict)
#     # axs[0].text(0.02, 0.02, f"File: {file_name.split('_')[-1]} px: {round(px_size, 5)}\nHV: {int(HT)} kV  Mag: {mag}kx", transform=axs[0].transAxes, fontdict=font_dict)
#     axs[0].add_artist(scalebar)

#         # Изменить текстовое отображение единицы на '1/nm'
#     if original_units == '1/nm':
#         scalebar.label.set_text('1/nm')


#     if denoised_image.ndim != 2:
#         denoised_picture = denoised_image[0,:,:,0]
#     else:
#         denoised_picture = denoised_image
#     axs[1].imshow(denoised_picture, 
#                     # extent=image_extent, 
#                     cmap=cmap)
#     axs[1].set_title(f'Denoised ({net_name})')
#     # axis off
#     axs[0].axis('off')
#     axs[1].axis('off')
#     # tight layout
#     fig.tight_layout(pad=0.1)
#     # save the figure
#     fig.savefig(save_path +file_name + f'original_vs_denoised_{net_name}.png', 
#                 dpi=600, 
#                 format='png', 
#                 bbox_inches='tight', 
#                 pad_inches=0)
    
#     if show==True:
#         plt.show()
#     plt.close()

#! new version of the function to plot original_vs_denoised images as from 06/09/2024
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

def plot_original_vs_denoised(original_image, 
                                denoised_image, 
                                net_name,
                                file_name, 
                                cmap,
                                save_path,
                                show=True, 
                                scale=1,
                                units='nm',
                                px_size=1,
                                HT='add HT here',
                                mag='add mag here'):
    # add scale
    image_scale_x = image_scale_y = scale

    image_FOV_x = original_image.data.shape[1] * image_scale_x
    image_FOV_y = original_image.data.shape[0] * image_scale_y
    image_extent = [0, image_FOV_x, 0, image_FOV_y]
    px_size = image_scale_x

    # Check if we are dealing with diffraction images in reciprocal space
    original_units = units
    if units == '1/nm':
        # Convert units to 'nm' for the ScaleBar, but we won't display 'nm'
        units = 'nm'
        Mag_CL = 'CL'
        mag_CL_unit = ''
        
        # Custom label formatter to keep the value but append '1/nm'
        def custom_label_formatter(value, unit):
            # Keep the scale value and append '1/nm'
            return f'{value:.2f} 1/nm'
    else:
        custom_label_formatter = None  # Use default formatting when not diffraction
        Mag_CL = 'Mag'
        mag_CL_unit = 'kx'

    # Create the scalebar
    scalebar = ScaleBar(
        dx=px_size, 
        units=units,
        location='lower right',
        frameon=True,
        scale_loc='top',
        length_fraction=0.15,
        height_fraction=0.005,
        label_formatter=custom_label_formatter  # Apply custom formatter to show value and '1/nm'
    )
    
    # Add magnification and pixel size to plot
    font_dict = {'size': 6,
                'color': 'white',
                'weight': 'bold',
                'verticalalignment': 'bottom'
                }

    fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), sharex=True, sharey=True)
    
    if original_image.ndim != 2:
        picture = original_image[0, :, :, 0]
    else:
        picture = original_image
    axs[0].imshow(picture, cmap=cmap)
    axs[0].set_title('Original')

    # Add file details, pixel size, and magnification
    axs[0].text(0.02, 0.02, 
                f"File: {file_name.split(' ')[0]} px: {round(px_size, 5)}\nHV: {int(HT)} kV  {Mag_CL} {mag}{mag_CL_unit}", 
                transform=axs[0].transAxes, 
                fontdict=font_dict)
    
    axs[0].add_artist(scalebar)

    if denoised_image.ndim != 2:
        denoised_picture = denoised_image[0, :, :, 0]
    else:
        denoised_picture = denoised_image
    axs[1].imshow(denoised_picture, cmap=cmap)
    axs[1].set_title(f'Denoised ({net_name})')
    
    # Turn off axis labels and ticks
    axs[0].axis('off')
    axs[1].axis('off')

    # Adjust layout
    fig.tight_layout(pad=0.1)
    
    # Save the figure
    fig.savefig(save_path + file_name + f'original_vs_denoised_{net_name}.png', 
                dpi=600, 
                format='png', 
                bbox_inches='tight', 
                pad_inches=0)
    
    if show:
        plt.show()
    
    plt.close()

# save denoised image files as tif and png files with scalebars, HT and magnification information
def save_tif_png_16bit_image_with_scalebar(image, 
                    save_path_tif16bit, 
                    save_path_png, 
                    file_name='put file name here',
                    px_size=1, 
                    mag=1, 
                    HT=1,
                    units='nm',
                    show=True):
    """
    Save image files as tif and png files with scalebars, HT and magnification information
    This function is used to save images as 16-bit TIFF and 8-bit PNG images. It uses the matplotlib library to create the image, and the ScaleBar function to add a scalebar. It also adds the magnification and pixel size to the image. This function is called by the  save_images function. 
    Args:
        image: Image to save.
        save_path_tif16bit: Path to the output directory for the 16-bit TIFF image.
        save_path_png: Path to the output directory for the 8-bit PNG image.
        file_name: Name of the image file.
        px_size: Pixel size in nm.
        mag: Magnification.
        HT: HT voltage.
        units: Unit of pixel size.
        show: Show the image.

    """
    fig, ax = plt.subplots()
    if image.ndim != 2:
        picture = image[0,:,:,0]
    else:
        picture = image
    ax.imshow(picture, cmap='gray')

    # Check if we are dealing with diffraction images in reciprocal space
    original_units = units
    if units == '1/nm':
        # Convert units to 'nm' for the ScaleBar, but we won't display 'nm'
        units = 'nm'
        Mag_CL = 'CL'
        mag_CL_unit = ''
        
        # Custom label formatter to keep the value but append '1/nm'
        def custom_label_formatter(value, unit):
            # Keep the scale value and append '1/nm'
            return f'{value:.2f} 1/nm'
    else:
        custom_label_formatter = None  # Use default formatting when not diffraction
        Mag_CL = 'Mag'
        mag_CL_unit = 'kx'

    scalebar = ScaleBar(
                    px_size, 
                    units=units,
                    location='lower right',
                    frameon=True,
                    scale_loc='top',
                    length_fraction='0.2',
                    height_fraction='0.015',
                    label_formatter=custom_label_formatter  # Apply custom formatter to show value and '1/nm'
                    )
    ax.add_artist(scalebar)
    # Remove the axis
    ax.axis('off')

    # Add magnification and pixel size to plot using .bcf file
    font_dict = {'size': 11,
                'color': 'white',
                'weight': 'bold',
                'verticalalignment': 'bottom'
                }
    #! set data zone text
    ax.text(0.02, 0.02, f"File name: {file_name.split(' ')[0]} px: {round(px_size, 5)}\nHV: {int(HT)} kV {Mag_CL} {mag}{mag_CL_unit}", transform=ax.transAxes, fontdict=font_dict)
    # ax.text(0.02, 0.02, f"File name: {file_name.split('_')[-2]} px: {round(px_size, 5)}\nHV: {int(HT)} kV  Mag: {mag}kx", transform=ax.transAxes, fontdict=font_dict)

    # Set the DPI value for the saved image
    dpi = 600 # Change this to the desired DPI value
    # Save the plot as a TIFF file
    fig.savefig(save_path_tif16bit+file_name+'.tif',
                dpi=dpi,
                format='tiff',
                bbox_inches='tight',
                pad_inches=0,
                )
    print('Saved as tiff: ', file_name)
    fig.savefig(save_path_png+file_name+'.png',
                dpi=dpi,
                format='png',
                bbox_inches='tight',
                pad_inches=0
                )
    print('Saved as png: ', file_name)
    if show==True:
        plt.show()
    plt.close()

# check the files if it is .tif or .emi file, load it and retirn the image as np.array
def check_and_load_image(
                        file_path, 
                        file_data_0, 
                        file_data, 
                        how_to_process='hyperspy' #! other option is tif
                        ): # file_data_0 is the part of the list file that have the metadata
    # if file_path.rsplit('.', maxsplit=1)[-1] == 'emi':
    if how_to_process == 'hyperspy':
        
        print(f'File {file_path} is not a TIFF image. It is .emi file. Using the hyperspy to load the data...')
        # s = hs.load(file)
        # image = s.data
        if file_data.data.ndim > 2:
            print(f'ndim > 2. Using the first image from the stack...')
            scale  = file_data.axes_manager[1].scale
            unit = file_data.axes_manager[1].units
            if unit=='µm':
                scale = scale*10**3
                unit = 'nm'
            # remove spaces
            unit = unit.replace(' ', '')
        else:
            print(f'ndim = 2. Using the image...')
            scale  = file_data.axes_manager[0].scale
            unit = file_data.axes_manager[0].units
            if unit=='µm':
                scale = scale*10**3
                unit = 'nm'
            unit = unit.replace(' ', '')
        px_size = scale

        # check if unit is in nm or 1/nm
        if unit == 'nm':
            image_mode = 'image'
            mag_item = file_data_0.original_metadata.get_item('ObjectInfo.ExperimentalDescription.Magnification_x')
        elif unit == '1/nm':
            image_mode = 'diffraction'
            mag_item = file_data_0.original_metadata.get_item('ObjectInfo.ExperimentalDescription.Camera length_m')
        else:
            print(f'Unit {unit} is not recognized. Exiting the script...')
            #! check
            sys.exit()
            # exit()
        

        image = file_data.data
        if image.ndim > 2:
            image = image[0,:,:]
        # aplly image adjustment
        image_adj = adjust_data_range(image, 
                                    p_low=0.5, 
                                    p_high=99.8)    
        # read the pixel size, magnnification and HT value from the file
        # px_size = file_data_0.original_metadata.get_item('ser_header_parameters.CalibrationDeltaX')*10**9 #! new: changed to check px_size and scale are the same
        # get magnification from the file
        # mag_item = file_data_0.original_metadata.get_item('ObjectInfo.ExperimentalDescription.Magnification_x') #! new: changed to check px_size and scale are the same
        if mag_item is not None and image_mode == 'image':
            mag = mag_item / 1000
        elif mag_item is not None and image_mode == 'diffraction':
            mag = f'{mag_item:.2e} m'
        else:
            mag = 1  # or any default value
            print("Warning: Magnification_x is not found in the metadata. Using default value.")
        # get HT from the file
        HT = file_data_0.original_metadata.get_item('ObjectInfo.ExperimentalDescription.High tension_kV')
        # unit = 'nm' #! new: changed to check px_size and scale are the same
        

    # elif file_path.rsplit('.', maxsplit=1)[-1] == 'tif':
    else:
        print(f'File {file_path} is a TIFF image. Using the tifffile to load the data...')
        # flag_tif = True
        # image = tifffile.imread(file_data)
        image = tifffile.imread(file_path)

        # This is a cell to read out the pixel size and unit of ImageJ TIFF files, may be changed depending on image metadata
        with tifffile.TiffFile(file_data) as tiff:
            tags = tiff.pages[0].tags
            px_size = tags['XResolution'].value[1]/tags['XResolution'].value[0]
            mag = None
            HT = None
            #tags["ResolutionUnit"].value
            if tiff.is_imagej:
                unit = tiff.imagej_metadata["unit"]
            else:
                unit = 'px'
        
    # else:
    #     print(f'File {f} is not a TIFF image. It is .emi file. Exiting the script...')
    #     # exit()
    #     continue
    return image, image_adj, px_size, mag, HT, unit, scale, image_mode

def apply_fft_and_plot(file_name, 
                        image, 
                        denoised_image,
                        scale, 
                        net_name, 
                        save_path_png,
                        save_path_fft_comparison,
                        show=True,
                        save_fft=True,
                        save_fft_comparison=True,
                        ):
    if image.ndim != 2:
        picture = image[0,:,:,0]
    else:
        picture = image
    
    if denoised_image.ndim != 2:
        denoised_picture = denoised_image[0,:,:,0]
    else:
        denoised_picture = denoised_image
    
    s_original = hs.signals.Signal2D(picture)
    s_denoised = hs.signals.Signal2D(denoised_picture)
    # add info about units to axes manager
    s_original.axes_manager[0].units = 'nm'
    s_original.axes_manager[1].units = 'nm'
    s_denoised.axes_manager[0].units = 'nm'
    s_denoised.axes_manager[1].units = 'nm'
    # add info about scale to axes manager
    s_original.axes_manager[0].scale = scale
    s_original.axes_manager[1].scale = scale
    s_denoised.axes_manager[0].scale = scale
    s_denoised.axes_manager[1].scale = scale


    # fft
    s_original_fft = s_original.fft(shift=True, apodization=True)
    s_denoised_fft = s_denoised.fft(shift=True, apodization=True)

    # 1 x 2 plot 
    fig, axs = plt.subplots(2, 2, figsize=(12,12), sharex=True, sharey=True)
    cmap= 'gray'
    axs[0][0].imshow(np.log10(np.abs(s_original_fft.data)**2+1), cmap=cmap)
    axs[0][0].set_title(f'Original {file_name.split(" ")[0]}')
    axs[0][1].imshow(np.log10(np.abs(s_denoised_fft.data)**2+1), cmap=cmap)
    axs[0][1].set_title(f'Denoised {file_name.split(" ")[0]}({net_name})')
    axs[1][0].imshow(picture, cmap=cmap)
    # axs[0].set_title('Original')
    axs[1][1].imshow(denoised_picture, cmap=cmap)
    #tick off
    axs[0][0].axis('off')
    axs[0][1].axis('off')
    axs[1][0].axis('off')
    axs[1][1].axis('off')
    # tight layout
    fig.tight_layout(pad=0.1)

    # axs[1].set_title(f'Denoised ({net_name})')
    if show==True:
        plt.show()

    # save fff of original image as a png file
    if save_fft == True:
        plt.imsave(save_path_png+file_name+'_fft_original.png',
                    np.log10(np.abs(s_original_fft.data)**2+1).astype('float16'),
                    cmap=cmap,
                    dpi=600,
                    format='png',
                    )
        #TODO
        # tifffile.imwrite(save_path_png+file_name+'_fft_original.tif', 
        #                 np.log10(np.abs(s_original_fft.data)**2+1).astype('float32'), 
        #                 imagej=True, 
        #                 resolution=(s_original_fft.axes_manager[0].scale, s_original_fft.axes_manager[1].scale),
        #                 metadata={'unit': '1/nm'})
    if save_fft_comparison == True:
        # save comparison plot of 2 images and corresponding fft as a png file
        fig.savefig(save_path_fft_comparison+file_name+'_fft.png',
                    dpi=600,
                    format='png',
                    bbox_inches='tight',
                    pad_inches=0
                    )
    plt.close()

# ? not sure if this part works since I have added the gpu_id to the function
def fcn_set_gpu_id(gpu_visible_devices: str = "0") -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_visible_devices

# denoising function
def denoise_image(image, net, nx, ny, max_size_gpu=1024):
    # ! Denoise
    if nx > max_size_gpu or ny > max_size_gpu:
        print('Patch-based denoise...')
        denoised = net.predict_patch_based(image, patch_size=int(max_size_gpu/2), stride=int(max_size_gpu/4), batch_size=2)
    else:
        denoised = net.predict(image)
    return denoised

def plot_image_fft_fft_cut(image,
                            file_name,
                            units, 
                            cut_freq,
                            scale,
                            HT,
                            mag,
                            px_size, #! new
                            apodization = 'Tukey', # better = 'Tukey', or True, or'hann' or 'hamming
                            cmap = 'inferno',
                            show=True,
                            save=True,
                            save_path = 'add path here',
                                    
                            ):
    
    # check image dimensions since there is sometime more than 2
    if image.ndim != 2:
        image = image[0,:,:,0]
    else:
        image = image
    
    # convert the image to hyperspy signal
    image = hs.signals.Signal2D(image)
    # add info about units to axes manager
    image.axes_manager[0].units = units
    image.axes_manager[1].units = units
    # add info about scale to axes manager in case if needed to cut the fft
    image.axes_manager[0].scale = scale
    image.axes_manager[1].scale = scale

    # aplly fft
    image_fft = image.fft(
                            shift=True, 
                            apodization=apodization
                            )
    # image_fft_cut = image_fft.isig[-15.0:15.0, -15.0:15.0]
    image_fft_cut = image_fft.isig[-cut_freq:cut_freq, -cut_freq:cut_freq]

    # add scales to be able to plot wit matplotlib
    image_scale_x = image_scale_y = image.axes_manager[0].scale
    image_fft_scale_x = image_fft_scale_y = image_fft.axes_manager[0].scale

    image_FOV_x = image.data.shape[0] * image_scale_x
    image_FOV_y = image.data.shape[1] * image_scale_y

    image_fft_FOV_x = image_fft.data.shape[0] * image_fft_scale_x
    image_fft_FOV_y = image_fft.data.shape[1] * image_fft_scale_y

    image_fft_cut_FOV_x = image_fft_cut.data.shape[0] * image_fft_scale_x
    image_fft_cut_FOV_y = image_fft_cut.data.shape[1] * image_fft_scale_y
    
    image_extent = [0, image_FOV_x, 0, image_FOV_y]
    image_fft_extent = [-image_fft_FOV_x/2, image_fft_FOV_x/2, -image_fft_FOV_y/2, image_fft_FOV_y/2]
    image_fft_cut_extent = [-image_fft_cut_FOV_x/2, image_fft_cut_FOV_x/2, -image_fft_cut_FOV_y/2, image_fft_cut_FOV_y/2]
    #TODO: continue here

    # plot image 1 x 3 plot: original image, full fft, cut fft
    fig, axs = plt.subplots(1, 3,  
                            figsize=(7.5, 2.2),
                            dpi = 300 # ! here I put 300 dpi. So not sure if it will look good when it will saved with 600 dpi.. so mabe better to save with 300 dpi since it is not very important. I will not use the FFT for further analysis.
                            )
    fig.tight_layout(pad=0.1)
    #! set colormap
    cmap = cmap
    # cmap = 'inferno'
    # cmap = 'viridis'
    # cmap = 'Greys'

    #plot image
    axs[0].imshow(image.data, 
                    extent=image_extent, 
                    cmap=cmap)

    axs[0].set
    axs[0].set_title('Original image')
    # set x label
    axs[0].set_xlabel('x axis (nm)')
    # set y label
    axs[0].set_ylabel('y axis (nm)')
    # # add scale bar
    px_size = image_scale_x
    # # units = image.axes_manager[0].units
    # units = 'nm'
        # Check if we are dealing with diffraction images in reciprocal space
    original_units = units
    if units == '1/nm':
        # Convert units to 'nm' for the ScaleBar, but we won't display 'nm'
        units = 'nm'
        Mag_CL = 'CL'
        mag_CL_unit = ''
        
        # Custom label formatter to keep the value but append '1/nm'
        def custom_label_formatter(value, unit):
            # Keep the scale value and append '1/nm'
            return f'{value:.2f} 1/nm'
    else:
        custom_label_formatter = None  # Use default formatting when not diffraction
        Mag_CL = 'Mag'
        mag_CL_unit = 'kx'

    scalebar = ScaleBar(
                    dx=1, # 1 since they are already calibrated by image extent in imshow
                    units=units,
                    location='lower right',
                    frameon=True,
                    scale_loc='top',
                    length_fraction='0.15',
                    height_fraction='0.005',
                    label_formatter=custom_label_formatter  # Apply custom formatter to show value and '1/nm'
                    )
    # Add magnification and pixel size to plot using .bcf file
    font_dict = {'size': 6,
                'color': 'white',
                'weight': 'bold',
                'verticalalignment': 'bottom'
                }

    #! set data zone text
    axs[0].text(0.02, 0.02, f"File: {file_name.split(' ')[0]} px: {round(px_size, 5)}\nHV: {int(HT)} kV  {Mag_CL} {mag}{mag_CL_unit}", transform=axs[0].transAxes, fontdict=font_dict)
    # axs[0].text(0.02, 0.02, f"File: {file_name.split('_')[-1]} px: {round(px_size, 5)}\nHV: {int(HT)} kV  Mag: {mag}kx", transform=axs[0].transAxes, fontdict=font_dict)

    axs[0].add_artist(scalebar)
    axs[1].imshow(adjust_data_range(np.log10(np.abs(image_fft.data)**2+1), p_low=10, p_high=95),
                            extent=image_fft_extent, 
                            cmap=cmap)
    axs[1].set_title('Full FFT')
    # set x label
    axs[1].set_xlabel('x axis (1/nm)')
    # set y label
    axs[1].set_ylabel('y axis (1/nm)')
    axs[2].imshow(adjust_data_range(np.log10(np.abs(image_fft_cut.data)**2+1), p_low=10, p_high=95), 
                    extent=image_fft_cut_extent,
                    cmap=cmap)
    # set x label
    axs[2].set_xlabel('x axis (1/nm)')
    # set y label
    axs[2].set_ylabel('y axis (1/nm)')
    axs[2].set_title('FFT cut')
    # plt.subplots_adjust(wspace=0.2, hspace=0.1, left=0, right=1, top=0.1, bottom=1)

    for ax in axs.flatten():
        ax.tick_params(axis='both', labelsize=6, width=1)  # Font size for tick labels
    # chow or not to show
    if show == True:
        plt.show()
    # save or not to save
    if save == True:
        fig.savefig(save_path+file_name+'_fft_cut.png', 
                    dpi=600, 
                    format='png', 
                    bbox_inches='tight', 
                    pad_inches=0)
    plt.close()