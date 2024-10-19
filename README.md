# Slide Extractor from Video

This project provides a Python-based command-line tool that extracts individual slides from a screen recording (e.g., `.mov` or `.mp4` video file). The tool detects when the content of a slide changes and saves each new slide as an image file in a dedicated folder. It helps you break a recorded presentation or video into individual slides for easy access.

## Features
- Automatically detects changes between frames and extracts only the slides.
- Saves each slide as a `.jpg` image.
- Includes a prompt to avoid overwriting existing files in case of duplicate output folder names.
- Supports input files in `.mov`, `.mp4`, and other video formats supported by OpenCV.

## Prerequisites

Before running the tool, ensure you have the following installed:

- **Python 3.x** (required for running the script)
- **OpenCV for Python** (for video processing)
- **NumPy** (used for frame comparison)

You can install the required Python packages from the provided `requirements.txt` file.

### Installation of Required Libraries:
```bash
pip install -r requirements.txt
```

## Setup

### 1. Clone the repository to your local machine:

```bash
git clone https://github.com/dougie181/slide-extractor.git
cd slide-extractor
```

### 2. Create a Virtual Environment

It is recommended to create a virtual environment to isolate the project’s dependencies. Here’s how to create one:

- **macOS/Linux:**

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

 - **Windows:**

	```powershell
	python -m venv .venv
	.venv\Scripts\activate
	```

### 3. Install the Required Libraries

Once the virtual environment is activated, install the required libraries using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
This will install OpenCV, NumPy, and any other dependencies listed in the `requirements.txt` file.

### 4. Make the Script Executable (macOS/Linux Only)
```bash
chmod +x ./scripts/processMovie.py
```

## Usage

### 1. Running the Script

Once you have set up the environment, you can use the `processMovie.py` script to extract slides from your video. Here’s how to run the script:

- **macOS/Linux:**

	```bash
	./scripts/processMovie.py <path_to_your_video_file>
	```

- **Windows:** 

	```powershell
	python scripts/processMovie.py <path_to_your_video_file>
	```

For example, if your video file is located in `data/input/slides.mov`, you would run:
```bash
./scripts/processMovie.py data/input/slides.mov  # macOS/Linux
```

or
```powershell
python scripts/processMovie.py data/input/slides.mov  # Windows
```

### 2. Overwrite Warning

If a folder already exists for the extracted slides (e.g., slides_extracted_frames), the script will ask whether you want to overwrite the existing folder:
```bash
Directory 'slides_extracted_frames' already exists. Do you want to overwrite it? (y/n):
```
If you choose y, the contents will be overwritten. If you choose n, the script will exit without making any changes.

### 3. Output

The script creates a folder named `<video_name>_extracted_frames` *(where <video_name> is the name of your video file without the extension)* and stores the extracted slides there as `.jpg` images.

For example, if your video file is `slides.mov`, the slides will be saved in a folder called `slides_extracted_frames`.

### 4. Example command
```bash
./scripts/processMovie.py data/input/slides.mov
```
The above command will extract slides from the `slides.mov` video located in the data/input folder and store the output images in the `slides_extracted_frame`s folder.

### 4. Deactivating the Virtual Environment

Once you are done with the project, you can deactivate the virtual environment by running:
```bash
deactivate
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.