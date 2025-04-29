import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced preprocessing function that handles:
    - Both training data and API input
    - Streaming data with prediction metadata
    - Robust error handling
    - Consistent feature engineering
    """
    try:
        df = df.copy()
        logger.info(f"Starting preprocessing. Initial shape: {df.shape}")
        
        # ========== FEATURE ENGINEERING ==========
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        fare_medians = df.groupby('Pclass')['Fare'].median()
        df['Fare'] = df.apply(
            lambda row: fare_medians[row['Pclass']] if pd.isna(row['Fare']) else row['Fare'],
            axis=1
        )
        
        # ========== CABIN/DECK PROCESSING ==========
        if 'Cabin' in df.columns:
            df['Cabin'] = df['Cabin'].fillna('Unknown')
            df['Deck'] = df['Cabin'].str[0].replace({'U': 'Unknown', None: 'Unknown'})
        elif 'Deck' not in df.columns:
            df['Deck'] = 'Unknown'
        
        # ========== CATEGORICAL ENCODING ==========
        categorical_cols = ['Sex', 'Embarked', 'Deck']
        encoders = {} 
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
                

                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le  # Store encoder if needed
        
        cols_to_drop = [
            'PassengerId', 'Name', 'Ticket', 'Cabin',
            'timestamp', 'predicted_survived', 'predicted_probability'
        ]
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        
        # Ensure we don't drop target column if present
        if 'Survived' in df.columns:
            cols_to_drop = [col for col in cols_to_drop if col != 'Survived']
        
        # Final cleanup
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Validate we have required columns
        required_columns = {'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Deck'}
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns after preprocessing: {missing_cols}")
        
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def load_data() -> pd.DataFrame:
    """
    Enhanced data loading function with better error handling
    
    Returns:
        DataFrame containing loaded and preprocessed Titanic dataset
    """
    try:
        url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
        logger.info(f"Loading data from {url}")
        
        df = pd.read_csv(url)
        logger.info(f"Raw data loaded. Shape: {df.shape}")
        
        return preprocess_data(df)
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise