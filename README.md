# zFISHer

Advanced image processing for multiplexed sequential FISH (Fluorescence In Situ Hybridization), built for the Napari viewer.

`zFISHer` is a Python-based tool designed to streamline the analysis of 3D microscopy data, with a focus on multiplexed sequential FISH workflows. It provides an intuitive graphical user interface within the `napari` ecosystem for image registration, segmentation, and fluorescent spot (puncta) detection and analysis.

## Key Features

- **Napari Integration:** Built as a `napari` widget for interactive data visualization and analysis.
- **Session Management:** Create new projects, save, and load analysis sessions.
- **Image Registration:** Automated and manual tools for aligning images from different rounds or channels.
- **Nuclei Segmentation:** Includes algorithms for identifying and segmenting cell nuclei.
- **Puncta Detection:** Multiple algorithms (Local Maxima, Laplacian of Gaussian, etc.) for detecting and counting fluorescent puncta.
- **Puncta Editing:** An interactive editor for manually adding, deleting, and curating detected puncta.
- **Colocalization Analysis:** Tools to analyze the spatial relationship between different fluorescent signals.

## Installation

It is recommended to install and run `zFISHer` within a Conda environment.

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd zFISHer
    ```

2.  **Create and Activate Conda Environment:**
    This project is tested with Python 3.10. You can create an environment from the provided `environment.yml` file, which includes all necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate zFISHer
    ```

## Usage

To launch the `zFISHer` application, run `main.py` from the root directory of the project:

```bash
python main.py
```

This will launch the `napari` viewer with the `zFISHer` widget loaded and ready for use.

## Codebase Structure

-   `main.py`: The main entry point to launch the application.
-   `requirements.txt` / `environment.yml`: Dependency lists.
-   `zfisher/`: The main Python package.
    -   `core/`: Core scientific algorithms for registration, segmentation, and analysis.
    -   `ui/`: The `magicgui`-based user interface and `napari` widget components.
    -   `utils/`: Helper utilities, such as logging.
-   `_legacy/`: Contains older versions of the codebase for historical reference.

## Contributing

This project was developed by Seth Staller. At this time, it is not actively seeking external contributions. For questions or licensing inquiries, please contact Seth.Staller@STJUDE.ORG.

## License

All rights reserved. Contact St. Jude Children's Research Hospital for licensing details.