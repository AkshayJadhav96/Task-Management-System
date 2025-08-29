import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from scipy.stats import randint, uniform
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.exceptions import NotFittedError

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')

class AITaskManagementSystem:
    def __init__(self):
        self.task_classifier = None
        self.priority_predictor = None
        self.workload_balancer = None
        self.vectorizer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.duration_scaler = StandardScaler()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.priority_encoder = None
        self.duration_predictor = None
        self.task_forecast_model = None
        self.feature_names = None
        self.expected_tfidf_features = 1000

    def load_and_prepare_data(self, csv_file_path=None, data_string=None):
        """Week 1: Data Gathering & Structuring"""
        print("=== WEEK 1: DATA GATHERING & PREPROCESSING ===")
        if csv_file_path:
            print(f"Loading data from CSV file: {csv_file_path}")
            try:
                self.df = pd.read_csv(csv_file_path).sample(n=1000, random_state=42)
                print("✅ CSV file loaded successfully!")
            except FileNotFoundError:
                print(f"❌ Error: CSV file '{csv_file_path}' not found.")
                self.df = pd.DataFrame()
            except Exception as e:
                print(f"❌ Error loading CSV: {e}")
                self.df = pd.DataFrame()
        else:
            print("No data source provided. Creating sample dataset for demonstration...")
            self.df = pd.DataFrame()

        if self.df.empty:
            print("⚠️ No data available. Please provide a valid dataset.")
            return self.df

        try:
            self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
            self.df['due_date'] = pd.to_datetime(self.df['due_date'], errors='coerce')
            self.df['user_workload'] = pd.to_numeric(self.df['user_workload'], errors='coerce')
            self.df['user_behavior_score'] = pd.to_numeric(self.df['user_behavior_score'], errors='coerce')
            self.df['estimated_duration_min'] = pd.to_numeric(self.df['estimated_duration_min'], errors='coerce')
            if 'actual_duration_min' in self.df.columns:
                self.df['actual_duration_min'] = pd.to_numeric(self.df['actual_duration_min'], errors='coerce')
        except Exception as e:
            print(f"⚠️ Warning: Error in data type conversion: {e}")
            print("Proceeding with available data...")

        print(f"Dataset loaded with {len(self.df)} tasks")
        print(f"Columns: {list(self.df.columns)}")
        return self.df

    def exploratory_data_analysis(self):
        """Week 1: Exploratory Data Analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        if self.df.empty:
            print("⚠️ No data available for EDA.")
            return

        print("\nDataset Info:")
        print(self.df.info())

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        print("\nTask Categories Distribution:")
        print(self.df['task_category'].value_counts())
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='task_category', order=self.df['task_category'].value_counts().index)
        plt.title('Task Categories Distribution', fontsize=14)
        plt.xlabel('Task Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nPriority Levels Distribution:")
        print(self.df['priority_level'].value_counts())
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='priority_level', order=['Low', 'Medium', 'High'])
        plt.title('Priority Levels Distribution', fontsize=14)
        plt.xlabel('Priority Level', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.show()

        print("\nUser Roles Distribution:")
        print(self.df['user_role'].value_counts())
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='user_role', order=self.df['user_role'].value_counts().index)
        plt.title('User Roles Distribution', fontsize=14)
        plt.xlabel('User Role', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nCompletion Status Distribution:")
        print(self.df['completion_status'].value_counts())
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='completion_status', order=self.df['completion_status'].value_counts().index)
        plt.title('Completion Status Distribution', fontsize=14)
        plt.xlabel('Completion Status', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.tight_layout()
        plt.show()

        self.df['days_until_due'] = (self.df['due_date'] - self.df['created_at']).dt.days
        self.df['is_overdue'] = self.df['days_until_due'] < 0
        self.df['task_description_length'] = self.df['task_description'].str.len()

        print(f"\nAverage days until due: {self.df['days_until_due'].mean():.2f}")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='days_until_due', bins=30, kde=True)
        plt.title('Distribution of Days Until Due', fontsize=14)
        plt.xlabel('Days Until Due', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.show()

        print(f"Overdue tasks: {self.df['is_overdue'].sum()}")
        plt.figure(figsize=(8, 8))
        overdue_counts = self.df['is_overdue'].value_counts()
        labels = ['Overdue' if label else 'Not Overdue' for label in overdue_counts.index]
        plt.pie(overdue_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
        plt.title('Proportion of Overdue Tasks', fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='task_description_length', bins=30, kde=True)
        plt.title('Distribution of Task Description Length', fontsize=14)
        plt.xlabel('Task Description Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.show()

    def preprocess_text(self, text):
        """Week 1: NLP Preprocessing"""
        if pd.isna(text):
            return ""
        text = text.lower()
        tokens = word_tokenize(text)
        processed_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words
        ]
        return ' '.join(processed_tokens)

    def prepare_features(self):
        """Week 1-2: Feature Engineering"""
        print("\n=== WEEK 2: FEATURE ENGINEERING ===")
        if self.df.empty:
            print("⚠️ No data available for feature engineering.")
            return

        print("Applying NLP preprocessing...")
        self.df['processed_description'] = self.df['task_description'].apply(self.preprocess_text)

        vectorizer_path = 'tfidf_vectorizer.pkl'
        try:
            if os.path.exists(vectorizer_path):
                print(f"Loading existing TF-IDF vectorizer from {vectorizer_path}...")
                self.vectorizer = joblib.load(vectorizer_path)
                if not hasattr(self.vectorizer, 'idf_'):
                    raise NotFittedError("Loaded TF-IDF vectorizer is not fitted.")
                tfidf_features = self.vectorizer.transform(self.df['processed_description'])
                current_features = tfidf_features.shape[1]
                if current_features < self.expected_tfidf_features:
                    print(f"⚠️ TF-IDF features ({current_features}) less than expected ({self.expected_tfidf_features}). Padding with zeros...")
                    padding = csr_matrix((tfidf_features.shape[0], self.expected_tfidf_features - current_features))
                    tfidf_features = hstack([tfidf_features, padding])
                elif current_features > self.expected_tfidf_features:
                    print(f"⚠️ TF-IDF features ({current_features}) more than expected ({self.expected_tfidf_features}). Truncating...")
                    tfidf_features = tfidf_features[:, :self.expected_tfidf_features]
            else:
                raise FileNotFoundError("TF-IDF vectorizer file not found.")
        except (FileNotFoundError, NotFittedError, Exception) as e:
            print(f"⚠️ Issue with TF-IDF vectorizer ({e}). Creating and fitting new vectorizer...")
            self.vectorizer = TfidfVectorizer(max_features=self.expected_tfidf_features, ngram_range=(1, 2))
            tfidf_features = self.vectorizer.fit_transform(self.df['processed_description'])
            joblib.dump(self.vectorizer, vectorizer_path)
            print("✅ TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")
            current_features = tfidf_features.shape[1]
            if current_features < self.expected_tfidf_features:
                padding = csr_matrix((tfidf_features.shape[0], self.expected_tfidf_features - current_features))
                tfidf_features = hstack([tfidf_features, padding])

        feature_columns = [
            'user_workload', 'user_behavior_score', 'days_until_due',
            'task_description_length', 'user_role_encoded', 'completion_status_encoded',
            'task_category_encoded', 'priority_level_encoded'
        ]

        for col in ['user_role', 'completion_status', 'task_category', 'priority_level']:
            encoder_path = 'label_encoders.pkl'
            if os.path.exists(encoder_path) and not self.label_encoders.get(col):
                self.label_encoders = joblib.load(encoder_path)
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(self.df[col])
            else:
                try:
                    self.df[f'{col}_encoded'] = self.label_encoders[col].transform(self.df[col])
                except ValueError:
                    print(f"⚠️ Warning: Unseen categories in {col}. Mapping to default value.")
                    self.df[f'{col}_encoded'] = self.df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_ else 0
                    )

        traditional_features = self.df[feature_columns].copy()
        traditional_features.fillna({
            'user_workload': traditional_features['user_workload'].mean(),
            'user_behavior_score': traditional_features['user_behavior_score'].mean(),
            'days_until_due': traditional_features['days_until_due'].median(),
            'task_description_length': traditional_features['task_description_length'].median(),
            'user_role_encoded': traditional_features['user_role_encoded'].mode()[0],
            'completion_status_encoded': 0,
            'task_category_encoded': traditional_features['task_category_encoded'].mode()[0],
            'priority_level_encoded': traditional_features['priority_level_encoded'].mode()[0]
        }, inplace=True)

        scaler_path = 'feature_scaler.pkl'
        try:
            if os.path.exists(scaler_path):
                print(f"Loading existing feature scaler from {scaler_path}...")
                self.scaler = joblib.load(scaler_path)
                traditional_features_scaled = self.scaler.transform(traditional_features)
            else:
                raise FileNotFoundError("Feature scaler file not found.")
        except (FileNotFoundError, Exception) as e:
            print(f"⚠️ Issue with feature scaler ({e}). Creating and fitting new scaler...")
            traditional_features_scaled = self.scaler.fit_transform(traditional_features)
            joblib.dump(self.scaler, scaler_path)
            print("✅ Feature scaler saved as 'feature_scaler.pkl'")

        self.X = hstack([tfidf_features, traditional_features_scaled])
        self.feature_names = self.vectorizer.get_feature_names_out().tolist() + feature_columns
        print(f"Feature matrix shape: {self.X.shape}")
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("✅ Label encoders saved as 'label_encoders.pkl'")

    def train_task_classifier(self):
        """Week 2: Task Classification Model using SVM"""
        print("\n=== WEEK 2: TASK CLASSIFICATION ===")
        model_path = 'task_classifier_model.pkl'
        if os.path.exists(model_path):
            print(f"Loading existing task classifier model from {model_path}...")
            self.task_classifier = joblib.load(model_path)
            if hasattr(self.task_classifier, 'n_features_in_') and self.X.shape[1] != self.task_classifier.n_features_in_:
                print(f"⚠️ Warning: Feature count mismatch. Expected {self.task_classifier.n_features_in_}, got {self.X.shape[1]}. Retraining model...")
                self.task_classifier = None
            else:
                print("✅ Task classifier model loaded successfully!")
                return

        if self.df.empty or self.X.shape[0] == 0:
            print("⚠️ No data available for training task classifier.")
            return

        y = self.df['task_category']
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(
            self.X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_val, self.y_train_val, test_size=0.25, random_state=42, stratify=self.y_train_val
        )

        print("Training Support Vector Machine (SVM) model...")
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        cv_scores = cross_val_score(svm_model, self.X_train, self.y_train, cv=3, scoring='accuracy')
        print(f"  Cross-Validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        svm_model.fit(self.X_train, self.y_train)
        y_val_pred = svm_model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_val_pred)
        precision = precision_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_val, y_val_pred, average='weighted', zero_division=0)

        print("Validation Set Metrics:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")

        y_test_pred = svm_model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        print(f"Test Set Accuracy: {test_accuracy:.3f}")

        self.task_classifier = svm_model
        joblib.dump(self.task_classifier, model_path)
        print(f"\n✅ SVM model trained and saved as '{model_path}' with validation accuracy: {accuracy:.3f}")

    def train_priority_predictor(self):
        """Week 3: Priority Prediction with Random Forest and XGBoost"""
        print("\n=== WEEK 3: PRIORITY PREDICTION ===")
        model_path = 'priority_predictor_model.pkl'
        encoder_path = 'priority_encoder.pkl'
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            print(f"Loading existing priority predictor model from {model_path} and encoder from {encoder_path}...")
            self.priority_predictor = joblib.load(model_path)
            self.priority_encoder = joblib.load(encoder_path)
            if hasattr(self.priority_predictor, 'n_features_in_') and self.X.shape[1] != self.priority_predictor.n_features_in_:
                print(f"⚠️ Warning: Feature count mismatch for priority predictor. Expected {self.priority_predictor.n_features_in_}, got {self.X.shape[1]}. Retraining model...")
                self.priority_predictor = None
            else:
                print("✅ Priority predictor model and encoder loaded successfully!")
                return

        if self.df.empty or self.X.shape[0] == 0:
            print("⚠️ No data available for training priority predictor.")
            return

        priority_encoder = LabelEncoder()
        y_priority = priority_encoder.fit_transform(self.df['priority_level'])
        self.priority_encoder = priority_encoder

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
        )

        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        rf_priority = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_dist_rf = {
            'n_estimators': randint(50, 300),
            'max_depth': [10, 20, 30, None],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2'],
            'min_weight_fraction_leaf': uniform(0, 0.5)
        }

        xgb_priority = XGBClassifier(random_state=42, eval_metric='mlogloss')
        param_dist_xgb = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }

        n_iter = 15
        cv = 3
        total_iterations = n_iter * cv
        progress_bar = tqdm(total=total_iterations, desc="RandomizedSearchCV Progress (Random Forest)")

        class ProgressCallback:
            def __init__(self, progress_bar, total_iterations):
                self.progress_bar = progress_bar
                self.iteration = 0
                self.total_iterations = total_iterations
                self.start_time = datetime.now()

            def __call__(self, *args, **kwargs):
                self.iteration += 1
                self.progress_bar.update(1)
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                if self.iteration > 0:
                    avg_time_per_iter = elapsed_time / self.iteration
                    remaining_iters = self.total_iterations - self.iteration
                    estimated_time_left = remaining_iters * avg_time_per_iter
                    self.progress_bar.set_postfix({"Est. Time Left (s)": f"{estimated_time_left:.1f}"})
                if self.iteration >= self.total_iterations:
                    self.progress_bar.close()

        print("Performing RandomizedSearchCV for Random Forest...")
        random_search_rf = RandomizedSearchCV(
            estimator=rf_priority,
            param_distributions=param_dist_rf,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        random_search_rf.fit(X_train_balanced, y_train_balanced)
        progress_bar.close()

        print("Performing RandomizedSearchCV for XGBoost...")
        progress_bar_xgb = tqdm(total=total_iterations, desc="RandomizedSearchCV Progress (XGBoost)")
        random_search_xgb = RandomizedSearchCV(
            estimator=xgb_priority,
            param_distributions=param_dist_xgb,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_weighted',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        random_search_xgb.fit(X_train_balanced, y_train_balanced)
        progress_bar_xgb.close()

        rf_score = random_search_rf.best_score_
        xgb_score = random_search_xgb.best_score_
        print(f"Random Forest Best CV F1-Score: {rf_score:.3f}")
        print(f"XGBoost Best CV F1-Score: {xgb_score:.3f}")

        if xgb_score > rf_score:
            self.priority_predictor = random_search_xgb.best_estimator_
            print(f"Selected XGBoost with parameters: {random_search_xgb.best_params_}")
        else:
            self.priority_predictor = random_search_rf.best_estimator_
            print(f"Selected Random Forest with parameters: {random_search_rf.best_params_}")

        y_pred_priority = self.priority_predictor.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_priority)
        precision = precision_score(y_test, y_pred_priority, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_priority, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred_priority, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred_priority, average='macro', zero_division=0)
        report = classification_report(y_test, y_pred_priority, target_names=priority_encoder.classes_)

        print(f"\nTest Set Evaluation:")
        print(f"Priority Prediction Accuracy: {accuracy:.3f}")
        print(f"Weighted Precision: {precision:.3f}")
        print(f"Weighted Recall: {recall:.3f}")
        print(f"Weighted F1-Score: {f1:.3f}")
        print(f"Macro F1-Score: {f1_macro:.3f}")
        print("Classification Report:\n", report)

        if hasattr(self.priority_predictor, 'feature_importances_'):
            feature_names = self.feature_names
            importances = self.priority_predictor.feature_importances_
            feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 Feature Importances:")
            for feature, importance in feature_importance:
                print(f"{feature}: {importance:.4f}")

        joblib.dump(self.priority_predictor, model_path)
        joblib.dump(self.priority_encoder, encoder_path)
        print("✅ Priority predictor model and encoder saved as 'priority_predictor_model.pkl' and 'priority_encoder.pkl'")

    def forecast_task_volume(self, periods=30):
        """Week 3: Time Series Forecasting of Task Creation"""
        print("\n=== TIME SERIES FORECAST: FUTURE TASK VOLUME ===")
        model_path = 'task_forecast_model.pkl'
        if os.path.exists(model_path):
            print(f"Loading existing forecast model from {model_path}...")
            self.task_forecast_model = joblib.load(model_path)
            print("✅ Forecast model loaded successfully!")
            return

        if self.df.empty or 'created_at' not in self.df.columns:
            print("⚠ No valid data available for forecasting.")
            return

        self.df['created_at'] = pd.to_datetime(self.df['created_at'], errors='coerce')
        self.df = self.df.dropna(subset=['created_at'])

        date_min = self.df['created_at'].min().date()
        date_max = self.df['created_at'].max().date()
        all_dates = pd.date_range(start=date_min, end=date_max, freq='D')
        daily_tasks = self.df.groupby(self.df['created_at'].dt.date).size().reset_index(name='task_count')
        daily_tasks.columns = ['ds', 'y']
        daily_tasks['ds'] = pd.to_datetime(daily_tasks['ds'])

        all_dates_df = pd.DataFrame({'ds': all_dates})
        daily_tasks = all_dates_df.merge(daily_tasks, on='ds', how='left').fillna({'y': 0})

        if len(daily_tasks) < 2:
            print("⚠ Not enough daily data for time series forecasting.")
            return

        train_size = int(0.8 * len(daily_tasks))
        train_data = daily_tasks.iloc[:train_size]
        test_data = daily_tasks.iloc[train_size:]

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        model.fit(train_data)

        if len(test_data) > 0:
            future_test = model.make_future_dataframe(periods=len(test_data), include_history=False)
            forecast_test = model.predict(future_test)
            forecast_test['yhat'] = forecast_test['yhat'].clip(lower=0)
            mae = mean_absolute_error(test_data['y'], forecast_test['yhat'][:len(test_data)])
            rmse = np.sqrt(mean_squared_error(test_data['y'], forecast_test['yhat'][:len(test_data)]))
            print(f"Test Set MAE: {mae:.2f} tasks")
            print(f"Test Set RMSE: {rmse:.2f} tasks")

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)

        print(f"\n📈 Forecasting task volume for the next {periods} days...")
        print(f"Average Predicted Tasks per Day: {forecast['yhat'][-periods:].mean():.2f}")
        fig = model.plot(forecast)
        plt.title("Forecasted Task Creation Volume")
        plt.xlabel("Date")
        plt.ylabel("Number of Tasks")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        model.plot_components(forecast)
        plt.tight_layout()
        plt.show()

        self.task_forecast_model = model
        self.task_forecast = forecast
        joblib.dump(self.task_forecast_model, model_path)
        print("✅ Forecast model saved as 'task_forecast_model.pkl'")

    def implement_workload_balancer(self):
        """Week 3: Workload Balancing Logic"""
        print("\n=== WORKLOAD BALANCING ===")
        if self.df.empty or 'user_id' not in self.df.columns:
            print("⚠ No user data available for workload balancing.")
            return

        user_stats = self.df.groupby('user_id').agg({
            'user_workload': 'first',
            'user_behavior_score': 'first',
            'task_id': 'count'
        }).rename(columns={'task_id': 'task_count'})
        user_stats.to_csv('users.csv')
        print("✅ User data saved as 'users.csv'")
        print("User Workload Statistics:")
        print(user_stats.head())

        def assign_task_to_user(task_priority, task_category, available_users):
            compatible_users = available_users[
                (available_users['user_workload'] < 15) &
                (available_users['user_behavior_score'] > 0.5)
            ]
            if len(compatible_users) == 0:
                return available_users.iloc[0]['user_id']
            selected_user = compatible_users.loc[compatible_users['user_workload'].idxmin()]
            return selected_user['user_id']

        self.workload_balancer = assign_task_to_user
        print("Workload balancer implemented successfully!")

    def predict_task_duration(self):
        """Week 3: Duration Prediction"""
        print("\n=== DURATION PREDICTION ===")
        model_path = 'duration_predictor_model.pkl'
        scaler_path = 'duration_scaler.pkl'
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Loading existing duration predictor model from {model_path} and scaler from {scaler_path}...")
            self.duration_predictor = joblib.load(model_path)
            self.duration_scaler = joblib.load(scaler_path)
            print("✅ Duration predictor model and scaler loaded successfully!")
            return

        duration_data = self.df[self.df['estimated_duration_min'].notna()].copy()
        if len(duration_data) < 10:
            print("⚠ Insufficient data for duration prediction training.")
            self.duration_predictor = None
            return

        Q1 = duration_data['estimated_duration_min'].quantile(0.25)
        Q3 = duration_data['estimated_duration_min'].quantile(0.75)
        IQR = Q3 - Q1
        duration_data = duration_data[
            (duration_data['estimated_duration_min'] >= Q1 - 1.5 * IQR) &
            (duration_data['estimated_duration_min'] <= Q3 + 1.5 * IQR)
        ]

        duration_features = [
            'user_workload', 'user_behavior_score', 'task_description_length',
            'task_category_encoded', 'priority_level_encoded'
        ]
        X_duration = duration_data[duration_features].copy()
        y_duration = pd.to_numeric(duration_data['estimated_duration_min'])

        X_duration.fillna({
            'user_workload': X_duration['user_workload'].mean(),
            'user_behavior_score': X_duration['user_behavior_score'].mean(),
            'task_description_length': X_duration['task_description_length'].median(),
            'task_category_encoded': X_duration['task_category_encoded'].mode()[0],
            'priority_level_encoded': X_duration['priority_level_encoded'].mode()[0]
        }, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_duration, y_duration, test_size=0.2, random_state=42
        )

        X_train_scaled = self.duration_scaler.fit_transform(X_train)
        X_test_scaled = self.duration_scaler.transform(X_test)

        param_dist = {
            'n_estimators': randint(50, 200),
            'max_depth': [10, 20, None],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5)
        }
        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=10, cv=3,
            scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
        )
        random_search.fit(X_train_scaled, y_train)

        self.duration_predictor = random_search.best_estimator_
        print(f"Best Parameters: {random_search.best_params_}")

        y_pred = self.duration_predictor.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Test Set MAE: {mae:.2f} minutes")
        print(f"Test Set RMSE: {rmse:.2f} minutes")
        print(f"Test Set R²: {r2:.3f}")

        joblib.dump(self.duration_predictor, model_path)
        joblib.dump(self.duration_scaler, scaler_path)
        print("✅ Duration predictor model and scaler saved as 'duration_predictor_model.pkl' and 'duration_scaler.pkl'")

    def create_dashboard_summary(self):
        """Week 4: Dashboard and Results"""
        print("\n=== WEEK 4: FINAL DASHBOARD SUMMARY ===")
        print("=== MODEL PERFORMANCE SUMMARY ===")
        if self.task_classifier and self.X.shape[1] == self.task_classifier.n_features_in_:
            y_true = self.df['task_category']
            y_pred_categories = self.task_classifier.predict(self.X)
            accuracy = accuracy_score(y_true, y_pred_categories)
            print(f"Task Classification Accuracy: {accuracy:.3f}")
        else:
            print("⚠ Cannot compute task classification accuracy: Model or feature mismatch.")

        if self.priority_predictor and self.X.shape[1] == self.priority_predictor.n_features_in_:
            y_true_priority = self.priority_encoder.transform(self.df['priority_level'])
            y_pred_priority = self.priority_predictor.predict(self.X)
            priority_accuracy = accuracy_score(y_true_priority, y_pred_priority)
            print(f"Priority Prediction Accuracy: {priority_accuracy:.3f}")
        else:
            print("⚠ Cannot compute priority prediction accuracy: Model or feature mismatch.")

        if self.duration_predictor:
            duration_data = self.df[self.df['estimated_duration_min'].notna()].copy()
            duration_features = [
                'user_workload', 'user_behavior_score', 'task_description_length',
                'task_category_encoded', 'priority_level_encoded'
            ]
            X_duration = duration_data[duration_features].fillna({
                'user_workload': duration_data['user_workload'].mean(),
                'user_behavior_score': duration_data['user_behavior_score'].mean(),
                'task_description_length': duration_data['task_description_length'].median(),
                'task_category_encoded': duration_data['task_category_encoded'].mode()[0],
                'priority_level_encoded': duration_data['priority_level_encoded'].mode()[0]
            })
            X_duration_scaled = self.duration_scaler.transform(X_duration)
            y_duration = pd.to_numeric(duration_data['estimated_duration_min'])
            y_pred_duration = self.duration_predictor.predict(X_duration_scaled)
            mae = mean_absolute_error(y_duration, y_pred_duration)
            print(f"Duration Prediction MAE: {mae:.2f} minutes")
        else:
            print("⚠ Cannot compute duration prediction MAE: Model not available.")

        print("\n=== SYSTEM STATISTICS ===")
        print(f"Total Tasks Processed: {len(self.df)}")
        print(f"Unique Users: {self.df['user_id'].nunique()}")
        print(f"Task Categories: {', '.join(self.df['task_category'].unique())}")
        print(f"Priority Levels: {', '.join(self.df['priority_level'].unique())}")

        print("\n=== USER PERFORMANCE ANALYSIS ===")
        user_performance = self.df.groupby('user_role').agg({
            'user_behavior_score': 'mean',
            'user_workload': 'mean',
            'task_id': 'count'
        }).round(3)
        print(user_performance)

    def predict_new_task(self, task_description, user_workload=5, user_behavior_score=0.8,
                        days_until_due=7, user_role='Developer'):
        """Predict category, priority, duration, and assign user for a new task"""
        print("\n=== NEW TASK PREDICTION ===")
        required_artifacts = {
            'vectorizer': 'tfidf_vectorizer.pkl',
            'label_encoders': 'label_encoders.pkl',
            'scaler': 'feature_scaler.pkl',
            'task_classifier': 'task_classifier_model.pkl',
            'priority_predictor': 'priority_predictor_model.pkl',
            'priority_encoder': 'priority_encoder.pkl',
            'duration_predictor': 'duration_predictor_model.pkl',
            'duration_scaler': 'duration_scaler.pkl'
        }
        for attr, path in required_artifacts.items():
            if not getattr(self, attr) and not os.path.exists(path):
                print(f"❌ Error: Required artifact '{path}' not found. Cannot make predictions.")
                return None, None, None, None
            if not getattr(self, attr):
                try:
                    setattr(self, attr, joblib.load(path))
                    print(f"Loaded {path}")
                except Exception as e:
                    print(f"❌ Error loading {path}: {e}")
                    return None, None, None, None

        try:
            if not hasattr(self.vectorizer, 'idf_'):
                print("❌ Error: TF-IDF vectorizer is not fitted. Cannot make predictions.")
                return None, None, None, None
        except AttributeError:
            print("❌ Error: TF-IDF vectorizer is not initialized. Cannot make predictions.")
            return None, None, None, None

        processed_desc = self.preprocess_text(task_description)
        try:
            tfidf_features = self.vectorizer.transform([processed_desc])
            current_features = tfidf_features.shape[1]
            if current_features < self.expected_tfidf_features:
                print(f"⚠ TF-IDF features ({current_features}) less than expected ({self.expected_tfidf_features}). Padding with zeros...")
                padding = csr_matrix((1, self.expected_tfidf_features - current_features))
                tfidf_features = hstack([tfidf_features, padding])
            elif current_features > self.expected_tfidf_features:
                print(f"⚠ TF-IDF features ({current_features}) more than expected ({self.expected_tfidf_features}). Truncating...")
                tfidf_features = tfidf_features[:, :self.expected_tfidf_features]
        except NotFittedError:
            print("❌ Error: TF-IDF vectorizer is not fitted during prediction.")
            return None, None, None, None

        # Use default values for encoded features initially
        user_role_encoded = self.label_encoders['user_role'].transform([user_role])[0] \
            if user_role in self.label_encoders['user_role'].classes_ else 0
        if user_role not in self.label_encoders['user_role'].classes_:
            print(f"⚠ Warning: User role '{user_role}' not seen during training. Using default encoding.")

        # Use mode of encoded features from training data
        task_category_encoded = self.df['task_category_encoded'].mode()[0] if 'task_category_encoded' in self.df else 0
        priority_level_encoded = self.df['priority_level_encoded'].mode()[0] if 'priority_level_encoded' in self.df else 0
        completion_status_encoded = 0  # Default for new tasks

        traditional_features = np.array([[
            user_workload,
            user_behavior_score,
            days_until_due,
            len(task_description),
            user_role_encoded,
            completion_status_encoded,
            task_category_encoded,
            priority_level_encoded
        ]])

        try:
            traditional_features_scaled = self.scaler.transform(traditional_features)
        except Exception as e:
            print(f"❌ Error scaling traditional features: {e}. Cannot make predictions.")
            return None, None, None, None

        X_new = hstack([tfidf_features, traditional_features_scaled])

        # Predict task category
        task_category_pred = 'Unknown'
        if self.task_classifier:
            expected_features = getattr(self.task_classifier, 'n_features_in_', self.X.shape[1] if hasattr(self, 'X') else 1008)
            if X_new.shape[1] == expected_features:
                task_category_pred = self.task_classifier.predict(X_new)[0]
            else:
                print(f"⚠ Feature mismatch for task classifier. Expected {expected_features}, got {X_new.shape[1]}. Using default category.")

        task_category_encoded = self.label_encoders['task_category'].transform([task_category_pred])[0] \
            if task_category_pred in self.label_encoders['task_category'].classes_ else 0
        if task_category_pred not in self.label_encoders['task_category'].classes_:
            print(f"⚠ Warning: Task category '{task_category_pred}' not seen during training. Using default encoding.")

        # Update traditional features with predicted task category
        traditional_features[0, 6] = task_category_encoded
        traditional_features_scaled = self.scaler.transform(traditional_features)
        X_new = hstack([tfidf_features, traditional_features_scaled])

        # Predict priority
        priority_pred = 'Unknown'
        priority_pred_encoded = 0
        if self.priority_predictor:
            expected_features = getattr(self.priority_predictor, 'n_features_in_', self.X.shape[1] if hasattr(self, 'X') else 1008)
            if X_new.shape[1] == expected_features:
                priority_pred_encoded = self.priority_predictor.predict(X_new)[0]
                priority_pred = self.priority_encoder.inverse_transform([priority_pred_encoded])[0]
            else:
                print(f"⚠ Feature mismatch for priority predictor. Expected {expected_features}, got {X_new.shape[1]}. Using default priority.")

        # Update traditional features with predicted priority
        traditional_features[0, 7] = priority_pred_encoded
        traditional_features_scaled = self.scaler.transform(traditional_features)
        X_new = hstack([tfidf_features, traditional_features_scaled])

        predicted_category = task_category_pred
        predicted_priority = priority_pred

        # Predict duration
        estimated_duration = None
        if self.duration_predictor:
            duration_features = np.array([[
                user_workload,
                user_behavior_score,
                len(task_description),
                task_category_encoded,
                priority_pred_encoded
            ]])
            try:
                duration_features_scaled = self.duration_scaler.transform(duration_features)
                estimated_duration = self.duration_predictor.predict(duration_features_scaled)[0]
            except Exception as e:
                print(f"⚠ Error predicting duration: {e}. Skipping duration prediction.")

        print(f"Task Description: {task_description}")
        print(f"Predicted Category: {predicted_category}")
        print(f"Predicted Priority: {predicted_priority}")
        if estimated_duration is not None:
            print(f"Predicted Estimated Duration: {estimated_duration:.1f} minutes")

        assigned_user = None
        if self.workload_balancer:
            try:
                available_users = pd.read_csv('users.csv') if os.path.exists('users.csv') else \
                    self.df[['user_id', 'user_workload', 'user_behavior_score']].drop_duplicates()
                assigned_user = self.workload_balancer(predicted_priority, predicted_category, available_users)
                print(f"🔄 Task Assigned To User ID: {assigned_user}")
            except Exception as e:
                print(f"⚠ Error loading user data: {e}. Cannot assign user.")
        else:
            print("⚠ Workload balancer not implemented. Cannot assign user.")

        return predicted_category, predicted_priority, estimated_duration, assigned_user

    def generate_task_forecast(self, periods=30):
        """Generate task volume forecast for a specified period"""
        if not self.task_forecast_model:
            model_path = 'task_forecast_model.pkl'
            if os.path.exists(model_path):
                try:
                    self.task_forecast_model = joblib.load(model_path)
                    print(f"Loaded forecast model from {model_path}")
                except Exception as e:
                    print(f"❌ Error loading {model_path}: {e}")
                    return None
            else:
                print("❌ Error: Forecast model not found.")
                return None
        future = self.task_forecast_model.make_future_dataframe(periods=periods)
        forecast = self.task_forecast_model.predict(future)
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def run_complete_project(csv_file_path=None):
    """Execute the complete 4-week project plan"""
    system = AITaskManagementSystem()
    print("🚀 STARTING AI-POWERED TASK MANAGEMENT SYSTEM PROJECT")
    print("=" * 60)

    if csv_file_path:
        print(f"📁 Loading data from CSV file: {csv_file_path}")
        df = system.load_and_prepare_data(csv_file_path=csv_file_path)
    else:
        print("📁 No CSV file provided. Using sample dataset for demonstration.")
        df = system.load_and_prepare_data()

    system.exploratory_data_analysis()
    system.prepare_features()
    system.train_task_classifier()
    system.train_priority_predictor()
    system.implement_workload_balancer()
    system.predict_task_duration()
    system.forecast_task_volume(periods=30)
    system.create_dashboard_summary()

    print("\n" + "="*50)
    print("TESTING NEW TASK PREDICTION")
    print("="*50)

    test_tasks = [
        "Fix critical bug in payment processing system",
        "Create user manual for new features",
        "Implement OAuth authentication",
        "Respond to customer inquiry about billing",
        "Write a linkedin post for challenges in task"
    ]

    for task in test_tasks:
        system.predict_new_task(task)
        print("-" * 40)

    return system

if __name__ == "__main__":
    ai_system = run_complete_project('task_dataset_noisy.csv')