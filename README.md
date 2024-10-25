# Electron Microscopy Image Denoising Project

**Author:** Dmitry Chezganov ([DexterChe on GitHub](https://github.com/DexterChe))  
**Date Created:** 25/10/2024

## Project Overview

This project provides a comprehensive toolkit for enhancing Electron Microscopy (EM) images by applying neural network-based denoising. It includes tools for both batch and single-image processing, making it ideal for handling large datasets as well as one-off images. The neural network used here was developed by Ivan Lobato, and details for installation can be found on his [GitHub repository](https://github.com/Ivanlh20/tk_r_em).

## Key Features

- **Batch Denoising**: Automated processing of large batches of EM images with customizable output directories for different file formats (16-bit, 32-bit TIFF, PNG).
- **Single Image Denoising**: Flexible, on-demand denoising for single images, useful for detailed analysis and review of specific EM images.
- **Flexible Output Options**: Saves images in multiple formats, allowing for both original and denoised versions, and includes additional FFT (Fast Fourier Transform) comparisons.
- **Comparison Outputs**: Generates side-by-side comparisons of denoised vs. original images and includes FFT outputs to assess denoising effectiveness.
- **Cross-Platform Compatibility**: Developed for macOS with Apple M chips, but can be easily adapted for Windows with CUDA-compatible GPUs.

## Folder Structure

The project includes a `comparison_denoised_vs_original` folder with comparisons of raw images and denoised versions, offering a visual assessment of the neural network’s effectiveness in noise reduction.

## Usage Instructions

### Prerequisites

- Install required Python libraries: `NN_denois_functions`, `hyperspy`, `tqdm`, and `gc`.
- Obtain and set up Ivan Lobato’s neural network by following the installation steps provided in his [GitHub repository](https://github.com/Ivanlh20/tk_r_em).

### Running the Scripts

1. **Batch Denoising**: Run the `Denoise_batch_if_list.ipynb` notebook to process a batch of EM images.
   - Customize the input and output directories as prompted in the notebook.
2. **Single Image Processing**: Use `Single_image_DM_version.ipynb` for single-image denoising, configuring paths for the input file and output folder.
3. **Outputs**: Denoised images, original images, and FFT comparisons are saved in organized subdirectories for easy retrieval.

### Notes

- This project was initially configured for macOS, but slight modifications will enable compatibility with Windows and CUDA-supported GPUs.

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](./LICENSE) file for details.

### Contact

For questions, suggestions, or contributions, please contact Dmitry Chezganov:  
- **GitHub**: [DexterChe](https://github.com/DexterChe)


