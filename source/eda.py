import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/churn.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (10, 6)

def save_plot(filename):
    """Helper to save plot and close figure to free memory"""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

def load_and_clean_data(path):
    print("⏳ Loading data...")
    df = pd.read_csv(path)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    df['ChurnNumeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df


def analyze_target_distribution(df):
    """Analyzes the balance of the target variable"""
    plt.figure(figsize=(6, 6))
    
    counts = df['Churn'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    plt.title('Churn Distribution (Target Balance)')
    save_plot("01_churn_distribution.png")

def analyze_numerical_features(df):
    """Analyzes Tenure, MonthlyCharges, TotalCharges"""
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    for col in num_cols:
        plt.figure(figsize=(10, 5))
        
        sns.kdeplot(data=df, x=col, hue="Churn", fill=True, common_norm=False, palette=['#2ecc71', '#e74c3c'], alpha=0.5)
        
        plt.title(f'Distribution of {col} by Churn Status')
        save_plot(f"02_dist_{col}.png")

def analyze_categorical_features(df):
    """Analyzes key categorical services and contracts"""
    cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'OnlineSecurity']
    
    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        
        ax = sns.countplot(x=col, data=df, hue='Churn', palette=['#2ecc71', '#e74c3c'])
        
        for container in ax.containers:
            ax.bar_label(container)
            
        plt.title(f'Churn Rates by {col}')
        plt.xlabel(col)
        plt.ylabel('Number of Customers')
        save_plot(f"03_cat_{col}.png")

def analyze_correlation(df):
    """Heatmap of numerical correlations"""
    plt.figure(figsize=(10, 8))
    
    numeric_df = df.select_dtypes(include=['number'])
    
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    save_plot("04_correlation_matrix.png")

def generate_text_report(df):
    """Saves summary statistics to a text file"""
    report_path = os.path.join(OUTPUT_DIR, "00_data_summary.txt")
    
    with open(report_path, "w") as f:
        f.write("DATASET SUMMARY REPORT\n")
        f.write("======================\n\n")
        
        f.write(f"Rows: {df.shape[0]}\n")
        f.write(f"Columns: {df.shape[1]}\n\n")
        
        f.write("MISSING VALUES:\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n\n")
        
        f.write("CHURN STATISTICS:\n")
        f.write(df['Churn'].value_counts(normalize=True).to_string())
        f.write("\n\n")
        
        f.write("NUMERICAL STATISTICS:\n")
        f.write(df.describe().to_string())

    print(f"✅ Saved text report: {report_path}")

if __name__ == "__main__":
    try:
        df = load_and_clean_data(DATA_PATH)
        
        print("📊 Generatng Visualizations...")
        analyze_target_distribution(df)
        analyze_numerical_features(df)
        analyze_categorical_features(df)
        analyze_correlation(df)
        generate_text_report(df)
        
        print(f"\n🎉 EDA Complete! Check the '{OUTPUT_DIR}' folder.")
        
    except FileNotFoundError:
        print("❌ Error: Data file not found. Please check 'data/' folder.")