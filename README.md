#  Python Practice Exercises

This repository contains a comprehensive collection of practice exercises designed to help you master fundamental data science libraries in Python, including **NumPy**, **Pandas**, **Matplotlib**, and **Scikit-Learn**.

Each notebook is structured with tasks and exercises, often building upon examples from introductory tutorials, to challenge and solidify your understanding of the library's core functionalities.

-----

## 🚀 Project Highlights: End-to-End Machine Learning
Based on the content of the notebooks, here are some of the observed results and the intended outcomes of the completed exercises for each practice module:

### 1. Scikit-Learn Exercises (`scikit-learn-exercises.ipynb`)

This notebook runs through a mini-machine learning project, focusing on two main problem types: Classification and Regression.

* **End-to-End Classification Workflow (on Heart Disease Data):**
    * **Data Preparation:** The initial steps successfully load and display the first few rows of the Heart Disease dataset, which includes features like `age`, `sex`, `cp` (chest pain type), `chol` (cholesterol), and a binary `target` variable.
    * **Model Evaluation:** After training, the notebook prepares to compare the performance of five different classification models: `LinearSVC`, `KNeighborsClassifier` (KNN), `SVC`, `LogisticRegression`, and `RandomForestClassifier`.
    * **Example Score:** One output snippet suggests an example model score of approximately **0.8316** (83.16% accuracy or similar metric) was achieved for one of the models tested.
    * **Prediction and Reporting:** The exercise includes steps to generate predictions (`y_preds`) on the test data and output a full **`classification_report`** (which includes precision, recall, and F1-score) to evaluate the model in detail.

* **Regression Problem and Model Persistence:**
    * **Metrics:** For the subsequent regression problem (e.g., predicting house prices), the notebook is set up to calculate standard regression evaluation metrics such as **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)**.
    * **Model Export:** A result of this workflow is the successful demonstration of **exporting the trained model** to a file (e.g., `heart-disease-random-forest-model.pkl` or similar) for later use.

### 2. Pandas Exercises (`pandas-exercises.ipynb`)

The focus is on fundamental data manipulation and structure creation:

* **Series Creation:** Separate Pandas **Series** are successfully created for different types of data, such as a Series of colors (e.g., Red, Blue, Green) and a Series of car types (e.g., BMW, Mercedes, GMC).
* **DataFrame Construction:** These Series are combined into a new **DataFrame (`car_data`)**.
* **Data Loading:** The notebook successfully loads a larger, practice DataFrame (implied to be a car sales dataset) which contains columns like `Make`, `Colour`, `Odometer (KM)`, `Doors`, and `Price`, confirming the ability to import data for analysis.

### 3. NumPy Exercises (`numpy-exercises.ipynb`)

This notebook confirms the basic setup for array manipulation:

* **Setup:** The successful import and use of the NumPy library under the alias `np` is established.
* **Initial Array Creation:** The first exercises involve the creation of basic NumPy data structures, such as a **1-dimensional array (`a1`)**.

### 4. Matplotlib Exercises (`matplotlib-exercises.ipynb`)

The notebook confirms the environment is ready for data visualization:

* **Setup:** Matplotlib's `pyplot` module is successfully imported as `plt`, and the environment is configured to display plots inline within the Jupyter notebook (`%matplotlib inline`).
* **Visualization Readiness:** The environment is confirmed to be capable of generating figures and axes, as evidenced by successful execution cells and instructions to explicitly reset figures using **`plt.subplots()`** for managing multiple visualizations.

## 🛠️ Setup and Installation

To run these notebooks locally, you will need a Python environment with the necessary packages installed.

### Prerequisites

  * **Python** (3.7+)
  * **Git** (for cloning the repository)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [YOUR_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment. The required packages are: `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.

    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required libraries
    pip install numpy pandas matplotlib scikit-learn 
    ```

-----

## 🤝 Contribution

Feel free to fork this repository, work through the exercises, and submit Pull Requests with suggested improvements, bug fixes, or alternative solutions.

Happy coding\! ✨
