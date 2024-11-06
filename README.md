---

# Violence and Weapon Detection

This repository provides a simple application to detect violent activities and weapons in video footage using deep learning models. The application is built using FastAPI, which serves the detection functionality through an API.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Model Setup](#model-setup)
- [Running the Application](#running-the-application)
- [API Usage](#api-usage)
- [License](#license)

## Project Overview

This project implements an API for detecting violent activities and weapons in video frames. It can be integrated into other systems for security monitoring purposes.

## Installation

To run this project, follow the instructions below.

### Prerequisites
- **Python 3.7+**
- **FastAPI** and **Uvicorn** (for running the API)

### Install Dependencies
1. Clone this repository:
   ```bash
   git clone https://github.com/helios190/Violence-and-Weapon-Detection.git
   cd Violence-and-Weapon-Detection
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Model Setup

1. **Download the Model**: Download the pre-trained model (link provided in the repository or as instructed).
2. **Create Model Directory**: In your project directory, create a new folder named `model`.
3. **Move Model File**: Place the downloaded model file into the `model` folder.

Your project structure should look like this:
```
Violence-and-Weapon-Detection/
├── app.py
├── model/
│   └── <model_file>
└── ...
```

## Running the Application

To start the application, simply use **Uvicorn**:

```bash
uvicorn app:app --reload
```

This will start the FastAPI server, accessible at `http://127.0.0.1:8000`.

## API Usage

You can test the endpoints by sending video frames for analysis. Further API usage instructions and endpoints can be found in the project documentation or by navigating to `http://127.0.0.1:8000/stream_detection` after starting the server.

## License

This project is licensed under the terms specified in the repository.

--- 

This README provides clear steps on setting up and running the project with `uvicorn`. Let me know if you need further customization!
