import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Configuration
plt.style.use('ggplot')
SAVE_FIGS = True  # Set to False if you don't want to save images
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and return Titanic dataset"""
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    return pd.read_csv(url)

def perform_eda(df: pd.DataFrame):
    """Perform exploratory data analysis with visualizations"""
    
    # Basic Data Inspection
    print("\n=== Data Overview ===")
    print("First 5 rows:")
    print(df.head())
    
    print("\nData Info:")
    print(df.info())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Survival Analysis
    plot_survival_distribution(df)
    plot_age_distribution(df)
    plot_fare_distribution(df)
    plot_class_survival(df)
    plot_gender_survival(df)
    
    # Feature Engineering & Advanced Analysis
    df_processed = preprocess_data(df.copy())
    plot_correlation_matrix(df_processed)
    
    return df_processed

def plot_survival_distribution(df):
    """Plot survival count distribution"""
    plt.figure(figsize=(8,6))
    ax = sns.countplot(x='Survived', data=df, color='grey')
    ax.set(title='Survival Count', xlabel='Survived', ylabel='Count')
    save_plot('survival_distribution.png')

def plot_age_distribution(df):
    """Plot age distribution with KDE"""
    plt.figure(figsize=(10,6))
    ax = sns.histplot(df['Age'], bins=30, kde=True, color='turquoise')
    ax.set(title='Age Distribution', xlabel='Age', ylabel='Frequency')
    save_plot('age_distribution.png')

def plot_fare_distribution(df):
    """Plot fare distribution with KDE"""
    plt.figure(figsize=(8,4))
    ax = sns.histplot(df['Fare'], bins=30, kde=True)
    ax.set(title="Fare Distribution", xlabel="Fare", ylabel="Frequency")
    save_plot('fare_distribution.png')

def plot_class_survival(df):
    """Plot survival by passenger class"""
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='Pclass', hue='Survived', data=df, 
                      palette=['darkblue', 'lime'])
    ax.set(title='Passenger Class vs Survival', xlabel='Class', ylabel='Count')
    save_plot('class_survival.png')

def plot_gender_survival(df):
    """Plot survival by gender"""
    plt.figure(figsize=(10,6))
    ax = sns.countplot(x='Sex', hue='Survived', data=df, 
                      palette=['red', 'darkgreen'])
    ax.set(title='Gender vs Survival', xlabel='Gender', ylabel='Count')
    save_plot('gender_survival.png')

def preprocess_data(df):
    """Preprocess data with feature engineering"""
    # Your original preprocessing steps
    df['Survived'] = df['Survived'].astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Label encoding
    for col in ['Sex', 'Embarked', 'Deck']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

def plot_correlation_matrix(df):
    """Plot feature correlation matrix"""
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    ax.set(title='Feature Correlation Matrix')
    save_plot('correlation_matrix.png')

def save_plot(filename):
    """Helper function to save plots"""
    if SAVE_FIGS:
        plt.savefig(FIG_DIR / filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Starting Titanic EDA...")
    raw_df = load_data()
    processed_df = perform_eda(raw_df)
    print("\n=== Processed Data Preview ===")
    print(processed_df.head())
    print("\nEDA complete! Check reports/figures for visualizations.")