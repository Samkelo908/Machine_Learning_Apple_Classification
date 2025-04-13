SVM Model Upgrade 🧠💻

🔍 Project Summary

As part of a recent hackathon, we were challenged to enhance the classification model built in Tutorials 1 through 4. We improved upon these earlier models and developed a highly optimized Support Vector Machine (SVM)-based classifier, which significantly boosts accuracy, performance, and usability. Below are the major improvements and comparisons.

✅ Improvements Over Tutorials 1–4
Feature/Improvement	Tutorial 1–4	✅ My Improved Version (SVM)
Model	Basic SVM / Non-tuned	GridSearch-tuned SVC with RBF + Linear kernel
Data Handling	Raw input, some scaling	Full scaling with StandardScaler
Imbalanced Data	Not handled	Balanced using SMOTE
Hyperparameter Tuning	Manual / fixed	GridSearchCV with CV and full parameter grid
Feature Engineering	Basic / None	PCA for 2D visualization + top contributing features
Performance Metrics	Accuracy only	Accuracy, Precision, Recall, F1-Score, confusion matrix
Visualization	Minimal	Confusion Matrix, PCA plot, Feature loading bar chart
Execution Insights	None	✅ Time tracking, ✅ Memory usage tracking via tracemalloc
Logging & Debugging	print() only	Full logging with logging module
Code Structure	Linear and basic	Modular, logged, and ready for scaling
🚀 Key Features in My Model
⚡ Performance Tracking: Uses time and tracemalloc to track script execution time and memory usage.

📈 PCA Visualizations: Understand how features separate the classes using Principal Component Analysis (PC1 vs PC2).


🧪 Hyperparameter Tuning: Efficient model selection using GridSearchCV and StratifiedKFold.


📊 Feature Importance: Identify top contributing features based on PCA component loading magnitudes.


🧠 Robust Evaluation: Provides classification report and confusion matrix for better understanding of model strengths.


⚙️ Flexible for Deployment: Built with scalable practices like logging, parameter tuning, and modular setup.


📊 Performance Metrics (Based on My Data)


Metric	Score
Accuracy	0.96
Precision	0.95
Recall	0.97
F1-Score	0.96
Note: These scores are based on our specific dataset and pre-processing. Actual performance may vary.

🔧 Technologies & Libraries

Python 🐍

Pandas, NumPy

Scikit-learn

imbalanced-learn (SMOTE)

Seaborn & Matplotlib (Visualization)

Logging & Tracemalloc (Monitoring)



📁 How to Run


Clone the repo

Place your dataset in the data/ folder

Edit the file_path in the script to point to your Excel file

Run the script:
python improved_svm.py

📌 Final Words

This SVM model was built to outperform the basic versions from our previous tutorials. It is faster, smarter, and more insightful, providing developers and data scientists with a robust tool for classifying data and gaining deeper insights.


🏆 We Won the Competition! 🧠💻


We are incredibly proud to share that Team Samkelo Maswana and Joshua Sutherland took home 1st place at our recent three-day machine learning competition! 🥇

This project marked a major milestone for us, as it was our first time diving into machine learning—and it wasn’t easy. From brainstorming ideas to debugging errors at 1 a.m., we faced every challenge head-on and pushed ourselves beyond what we thought we could do.


🔍 How We Did It


We built a machine learning model using the Support Vector Machine (SVM) algorithm—a powerful classifier used for tasks like pattern recognition and data prediction. Here's how we approached it:

Data Preparation: We started by cleaning and preparing our dataset, ensuring it was properly labeled and balanced.

Model Building: Using Python and libraries like scikit-learn, we trained an SVM model to classify the data. Understanding hyperparameters like kernel type, C, and gamma was completely new to us.

Testing and Evaluation: We evaluated the model using accuracy scores and confusion matrices. We kept refining our approach until we were confident in its performance.

Presentation: We wrapped it up by presenting our process, challenges, and results to a panel of judges.



💪 The Challenge


What made this victory even more rewarding was the struggle we overcame. Neither of us had ever written a line of machine learning code before. Learning the theory behind SVMs, experimenting with different kernels (like linear and RBF), and figuring out model tuning—all within three days—was intense.

There were moments where things didn’t work, models failed to converge, or predictions were off. But we never gave up. We asked questions, debugged, tested, and supported each other through every late-night sprint.


👥 Teamwork & Dedication


This wouldn’t have been possible without pure teamwork. Both of us—Samkelo Maswana and Joshua Sutherland —gave 100%. We shared tasks, kept the energy up, and always believed we could do it if we gave it our all.


🏁 The Result


Our hard work paid off. We not only completed the project on time—we won the competition. 🎉

We’ll be posting pictures of us working, coding, and presenting soon. This is only the beginning!

 Behind the Scenes – The Grind

 
 ![IMG-20250410-WA0041](https://github.com/user-attachments/assets/3b976d38-2662-4fac-87f1-ca8c30f5b3d9)

 ![IMG-20250410-WA0039](https://github.com/user-attachments/assets/456262c6-c26f-4c73-b056-48b947f1ecbd)

Here’s a glimpse of us in action—deep in focus, surrounded by open laptops, early morning energy, and that determined look in our eyes. This picture was taken in the morning while we were fully zoned in, troubleshooting our SVM model and refining our presentation slides. It wasn’t just about writing code—it was about learning together, encouraging each other, and staying motivated throughout the entire three-day challenge. These moments built the foundation for our success and made the win even more meaningful.


🎤 The Big Moment – Presenting Our Work


![IMG-20250410-WA0016](https://github.com/user-attachments/assets/41bebac8-ccbe-4bfc-9741-e9b8b0c3557c)

![IMG-20250410-WA0052](https://github.com/user-attachments/assets/6eea9268-f8c3-4396-863e-b44b053c8c86)

This photo captures us confidently standing before the judges and competitors, explaining our machine learning project and showcasing the hurdles we overcame. It was the final stretch, and despite the nerves, we spoke from the heart. The positive feedback we received made it all worth it.

