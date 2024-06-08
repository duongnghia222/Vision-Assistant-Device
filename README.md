# Vision Assistant Device

The objective of this project is to design and deploy an object detection device to assist individuals with visual impairments in their daily lives. This system applies computer vision techniques and image processing to help individuals recognize and locate objects and avoid obstacles.

#vosk #pyttx3 #pyrealsense2 #ultralytics #opencv

## Project Structure

The main components of the project are:

- **main.py**: The main entry point of the application.
- **test.py**: Contains various tests and examples of how to use the different components of the application.
- **tools/**: Contains utility scripts and classes such as [virtual_assistant.py](tools/virtual_assistant.py) which provides functionality for a virtual assistant, and `realsense_camera.py` for interfacing with a RealSense camera.
- **utils/**: Contains utility scripts and classes for the machine learning models.
- **models/**: Contains the trained machine learning models used in the application.
- **nltk_data/**: Contains data for the NLTK library used for natural language processing tasks.
- **pictures/**: Contains sample pictures for testing.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/duongnghia/Vision-Assistant-Device.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Vision-Assistant-Device
    ```
3. Install the required Python packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```


## Usage

To start the application, run the `main.py` script:
```bash
python main.py
```


## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please read the CONTRIBUTING guidelines for more information.


https://thanhhungqb.github.io/iaslab/research/

Demo

https://www.youtube.com/watch?v=G-P9G4ZW4Fg