# Exjobb Projects Visualizer

![Project Logo](./logo.png) 

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Scraper](#running-the-scraper)
  - [Launching the Streamlit App](#launching-the-streamlit-app)
- [Directory Structure](#directory-structure)
- [Automated Updates](#automated-updates)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project automates the process of scraping **Exjobb** (master thesis) project data from [Linköping University](https://exjobb.liu.se/en-US/) using Selenium. The scraped data is stored in a CSV file and visualized through an interactive Streamlit application. Additionally, GitHub Actions is configured to run the scraper daily, ensuring the data remains up-to-date.

## Features

- **Automated Web Scraping**: Uses Selenium to extract project details such as title, organization, research field, and application deadlines.
- **Data Storage**: Saves scraped data in a structured CSV format for easy access and analysis.
- **Interactive Visualization**: Streamlit app provides interactive charts and filters to explore the data.
- **Daily Updates**: GitHub Actions workflow ensures the scraper runs daily, keeping the dataset current.
- **Data Download**: Users can download filtered data in CSV and Excel formats directly from the Streamlit app.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.7 or higher** installed on your machine. You can download it from [here](https://www.python.org/downloads/).
- **Google Chrome** browser installed. Download it from [here](https://www.google.com/chrome/).
- **ChromeDriver** compatible with your Chrome version. You can download it from [here](https://chromedriver.chromium.org/downloads).

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/liu-exjobb-crawler.git
   cd liu-exjobb-crawler
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up ChromeDriver**

   - Download the ChromeDriver version that matches your installed Chrome browser.
   - Extract the `chromedriver` executable and place it in a directory that's in your system's `PATH` or specify its path in the `liu_data.py` script.

## Usage

### Running the Scraper

The scraper script `liu_data.py` uses Selenium to navigate the Exjobb website, extract project information, and save it to a CSV file.

```bash
python liu_data.py
```

After running, the scraped data will be available at `data/exjobb_projects.csv`.

### Launching the Streamlit App

The Streamlit application `streamlit_app.py` visualizes the scraped data.

```bash
streamlit run streamlit_app.py
```

This command will open a new tab in your default web browser displaying the interactive dashboard.

## Directory Structure

```
├── data
│   └── exjobb_projects.csv       # Scraped project data
├── liu_data.py                   # Web scraper script
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── streamlit_app.py              # Streamlit visualization app
```

- **data/**: Contains the CSV file with the scraped Exjobb project data.
- **liu_data.py**: Python script that performs web scraping using Selenium.
- **streamlit_app.py**: Streamlit application for data visualization and interaction.
- **requirements.txt**: Lists all Python libraries required to run the project.
- **README.md**: Provides an overview and instructions for the project.

## Automated Updates

To ensure that the scraped data is updated daily, a GitHub Actions workflow is set up.

1. **GitHub Actions Workflow**

   The workflow file `.github/workflows/auto.yml` is configured to run the `liu_data.py` script daily at 02:00 UTC.

2. **Setup Steps**

   - Ensure that `.github/workflows/auto.yml` is present in your repository.
   - The workflow installs necessary dependencies, Chrome, and ChromeDriver before running the scraper.
   - After scraping, it commits and pushes the updated `exjobb_projects.csv` back to the repository.

3. **Monitoring**

   - Navigate to the **Actions** tab in your GitHub repository to monitor workflow runs.
   - Ensure that the workflow completes successfully and updates the data as expected.

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Make Changes and Commit**

   ```bash
   git commit -m "Add new feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any inquiries or suggestions, please contact [me](mailto:djmax96945147@outlook.com).

---

🚀
