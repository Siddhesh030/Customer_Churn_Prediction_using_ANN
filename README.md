# 📊 Customer Churn Prediction Using Artificial Neural Network (ANN)

An end-to-end Deep Learning project that predicts customer churn using an **Artificial Neural Network (ANN)**. This project demonstrates how deep learning can help businesses identify customers who are likely to leave and support data-driven customer retention decisions through an interactive Streamlit web application.

## 🌐 Live Demo

🚀 **Try the application:** [Customer Churn Prediction App](https://customerchurnpredictionusingann-wgkg76ggijtdnksawappcsg.streamlit.app/)

💻 **GitHub Repository:** [Customer Churn Prediction Using ANN](https://github.com/Siddhesh030/Customer_Churn_Prediction_using_ANN)

Open the live application, enter the required customer information, and explore the churn prediction results.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#️-technology-stack)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Artificial Neural Network (ANN)](#-artificial-neural-network-ann)
- [Project Structure](#-project-structure)
- [Installation and Setup](#️-installation-and-setup)
- [Running the Application](#-running-the-application)
- [How to Use the Application](#-how-to-use-the-application)
- [Model Evaluation](#-model-evaluation)
- [Applications](#-real-world-applications)
- [Future Improvements](#-future-improvements)
- [Limitations](#️-limitations)
- [Author](#-author)

## 📖 Project Overview

Customer churn occurs when customers stop using a company's products or services. High churn rates can negatively affect revenue, customer relationships, and long-term business growth.

This project uses an **Artificial Neural Network (ANN)** to analyze customer information and predict whether a customer is likely to leave a business.

The trained model learns patterns from historical customer data and uses those patterns to generate predictions for new customer records. A Streamlit application provides an interactive interface for accessing the prediction system without requiring users to run the model manually.

The project combines deep learning, data preprocessing, model development, and web application deployment in one practical machine learning solution.

## 🎯 Problem Statement

Businesses often struggle to identify customers who are likely to discontinue their services before they actually leave.

Traditional approaches based on manual analysis may fail to recognize complex patterns across multiple customer attributes.

The goal of this project is to develop a predictive system that uses an Artificial Neural Network to classify customer churn and help businesses identify customers who may require additional attention.

## 🎯 Project Objectives

- Develop an ANN-based binary classification model for customer churn prediction.
- Prepare customer data for deep learning through appropriate preprocessing.
- Learn relationships between customer attributes and churn behavior.
- Generate predictions for individual customer records.
- Build an interactive web interface using Streamlit.
- Deploy the application online so users can access it through a browser.
- Demonstrate a practical application of deep learning in business analytics.

## ✨ Key Features

- **🤖 ANN-Based Prediction:** Uses an Artificial Neural Network to predict customer churn.
- **📊 Customer Data Analysis:** Uses customer attributes as model inputs.
- **⚙️ Data Preprocessing:** Supports the encoding and scaling procedures required by the trained model.
- **🔮 Interactive Predictions:** Allows users to submit customer information through a web interface.
- **🌐 Live Deployment:** Provides browser-based access through Streamlit.
- **🐍 Python-Based Implementation:** Uses the Python ecosystem for machine learning and application development.
- **💡 Business Relevance:** Demonstrates how churn predictions can support customer retention planning.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | Building and training the ANN, if used in the implementation |
| NumPy | Numerical operations |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | Data preprocessing and model evaluation |
| Streamlit | Interactive web application |
| Jupyter Notebook | Model experimentation and development, if included |
| Git and GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Web application deployment |

*Note: The exact libraries and tools depend on the implementation in the repository. Confirm the actual dependencies in `requirements.txt` before finalizing this list.*

## 🔄 Machine Learning Workflow

The project follows a typical deep learning classification workflow.

1. **Data Collection:** Obtain the historical customer dataset containing relevant customer attributes and churn labels.
2. **Data Exploration:** Inspect the dataset, understand its features, and identify patterns relevant to churn.
3. **Data Preprocessing:** Handle categorical and numerical variables using the transformations required by the model.
4. **Feature Preparation:** Separate the input features from the target variable.
5. **Train-Test Split:** Divide the available data into training and testing sets to evaluate generalization.
6. **Feature Scaling and Encoding:** Apply the fitted preprocessing transformations consistently to training data and prediction inputs.
7. **ANN Model Development:** Build a neural network suitable for binary classification.
8. **Model Training:** Train the network on labeled customer data.
9. **Model Evaluation:** Assess performance using suitable classification metrics.
10. **Application Integration:** Load the trained model and required preprocessing objects into the Streamlit application.
11. **Deployment:** Publish the application so users can access it online.

### Workflow Diagram

```text
Historical Customer Data
          |
          v
    Data Exploration
          |
          v
   Data Preprocessing
          |
          v
 Feature Encoding/Scaling
          |
          v
  Train-Test Split
          |
          v
 ANN Model Training
          |
          v
   Model Evaluation
          |
          v
 Save Model and Preprocessors
          |
          v
 Streamlit Application
          |
          v
 Customer Churn Prediction
```

## 🧠 Artificial Neural Network (ANN)

An Artificial Neural Network is a deep learning model inspired by interconnected neurons. It learns patterns in input data by adjusting weights during training.

For customer churn prediction, the ANN processes the prepared customer features and produces an output representing the predicted churn class or probability.

### Typical ANN Architecture

A binary classification ANN generally consists of the following components:

- **Input Layer:** Receives the processed customer features.
- **Hidden Layers:** Learn relationships and patterns in the customer data.
- **Activation Functions:** Introduce non-linearity so the network can learn complex patterns.
- **Output Layer:** Commonly uses a sigmoid activation function for binary classification.
- **Loss Function:** Binary cross-entropy is commonly used to measure prediction error.
- **Optimizer:** An optimizer such as Adam can update model weights during training.

The exact number of hidden layers, neurons, activation functions, and training parameters should match the architecture implemented in the project.

### Prediction Interpretation

The model's output can be interpreted as an estimate of the probability that a customer belongs to the churn class, provided the model is configured to produce probabilities.

For a binary classification threshold of 0.5, a typical interpretation is:

- Probability at or above 0.5: predicted churn.
- Probability below 0.5: predicted non-churn.

The actual threshold and the application's displayed output should be verified against the implementation.

## 📁 Project Structure

The following is an illustrative project structure. Adjust it to match the actual files in the repository.

```text
Customer_Churn_Prediction_using_ANN/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── *.ipynb                 # Model development notebooks, if included
├── *.csv                   # Dataset, if included
├── *.keras / *.h5          # Trained ANN model, if included
├── *.pkl                   # Preprocessing objects, if included
│
└── .gitignore              # Git ignore rules
```

**Important:** Replace the illustrative filenames with the real filenames in your repository. Do not add files to this section that are not present in your project.

## ⚙️ Installation and Setup

Follow these steps to download the project and run it on your local machine.

### Prerequisites

Make sure you have installed:

- Python 3.10 or another version supported by the project's dependencies.
- Git.
- A code editor such as Visual Studio Code.
- A web browser.

### Step 1: Clone the Repository

Open a terminal and execute:

```bash
git clone https://github.com/Siddhesh030/Customer_Churn_Prediction_using_ANN.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Customer_Churn_Prediction_using_ANN
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check the Python version and the package versions specified in `requirements.txt`.

## ▶️ Running the Application

After installing the dependencies, start the Streamlit application from the project directory.

```bash
streamlit run app.py
```

If your Streamlit entry-point file has a different name or is located in a subdirectory, use its actual path instead.

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser to interact with the application.

## 🖥️ How to Use the Application

You can use the project in two ways.

### Option 1: Use the Live Application

1. Open the [deployed Customer Churn Prediction application](https://customerchurnpredictionusingann-wgkg76ggijtdnksawappcsg.streamlit.app/).
2. Wait for the application to finish loading.
3. Enter the customer information requested by the interface.
4. Submit the information using the application's prediction control.
5. Review the resulting churn prediction.
6. If available, examine the predicted probability and other displayed metrics.

### Option 2: Run the Application Locally

1. Clone the GitHub repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies.
4. Ensure the required model and preprocessing files are available.
5. Run the Streamlit entry-point file.
6. Open the local application URL and provide customer inputs.

## 📈 Model Evaluation

Evaluating a classification model is important for understanding how well it distinguishes between customers who churn and those who remain.

The following metrics are useful for evaluating an ANN-based churn prediction model:

| Metric | Description |
|---|---|
| Accuracy | Proportion of all predictions that are correct |
| Precision | Proportion of predicted churn cases that are actual churn cases |
| Recall | Proportion of actual churn cases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |
| ROC-AUC | Measures the model's ability to rank churn cases above non-churn cases |

For customer churn prediction, recall can be particularly important because failing to identify a customer who is about to leave may result in a missed retention opportunity.

**Note:** Add your actual test-set metrics, confusion matrix, or training curves here once they have been measured. Do not report unverified accuracy or performance figures.

## 💼 Real-World Applications

Customer churn prediction can support decision-making in several industries:

- **Banking and Finance:** Identify customers who may close accounts or move to another provider.
- **Telecommunications:** Identify subscribers who may discontinue a mobile or internet service.
- **E-commerce:** Identify customers who may stop purchasing from a platform.
- **Subscription Services:** Identify subscribers who may cancel recurring plans.
- **Customer Relationship Management:** Support customer segmentation and retention campaigns.

These are potential applications of churn prediction rather than claims that the project has been tested in every industry.

## 🔮 Future Improvements

Potential enhancements include:

- Compare ANN performance with Logistic Regression, Random Forest, XGBoost, and other classification models.
- Perform hyperparameter tuning to improve generalization.
- Add interactive charts and customer analytics to the Streamlit dashboard.
- Display churn probabilities and configurable risk categories.
- Use explainability techniques to understand the factors influencing predictions.
- Add batch prediction for CSV uploads.
- Improve validation, error handling, and input checks.
- Track model performance and data drift over time.
- Add model evaluation visualizations and a downloadable prediction report.
- Improve application responsiveness and accessibility.

## ⚠️ Limitations

- Prediction quality depends on the quality and representativeness of the training dataset.
- The model may perform differently on customer populations that differ from the training data.
- A predicted churn probability is an estimate, not a guarantee that a customer will leave.
- Appropriate preprocessing must be applied consistently during training and inference.
- Predictions should support business decisions rather than replace human judgment.
- The model's reliability should be assessed using held-out data and relevant evaluation metrics.

## 📚 Learning Outcomes

This project provides practical experience with:

- Artificial Neural Networks and binary classification.
- Data cleaning and preprocessing.
- Feature encoding and scaling.
- Training and evaluating deep learning models.
- Integrating a trained model into a Streamlit application.
- Deploying a machine learning application online.
- Using GitHub to document and share a data science project.

## 👨‍💻 Author

**Siddhesh**

- GitHub: [@Siddhesh030](https://github.com/Siddhesh030)
- Project Repository: [Customer Churn Prediction using ANN](https://github.com/Siddhesh030/Customer_Churn_Prediction_using_ANN)
- Live Application: [Try the Project](https://customerchurnpredictionusingann-wgkg76ggijtdnksawappcsg.streamlit.app/)

## ⭐ Support

If you find this project useful or informative, consider giving the repository a ⭐ on GitHub.

Feedback, suggestions, and contributions are welcome!

---

**Built with Python, Deep Learning, Artificial Neural Networks, and Streamlit.**
