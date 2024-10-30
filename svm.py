import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline

def prepare_data(excel_path):
    xls = pd.ExcelFile(excel_path)
    X = []
    y = []

    for sheet_name in xls.sheet_names:
        if 'Interaction_Comparisons' in sheet_name:
            print(f"Processing sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name)

            # Keep rows where p-value is less than 0.05 (significant interaction)
            df['label'] = df['p-unc'].apply(lambda p: 1 if p < 0.05 else 0)

            # Append features and labels
            X.extend(df[['A', 'B', 'p-unc']].values)  # Using A, B, and p-unc as features
            y.extend(df['label'].values)

    X_df = pd.DataFrame(X, columns=['A', 'B', 'p-unc'])

    # Encode categorical columns 'A' and 'B' (like 'hispanic', 'non-hispanic') to numeric values
    label_encoder = LabelEncoder()
    X_df['A'] = label_encoder.fit_transform(X_df['A'].astype(str))
    X_df['B'] = label_encoder.fit_transform(X_df['B'].astype(str))

    return X_df, pd.Series(y)

def run_svm_analysis(excel_path):
    # Prepare the data
    X, y = prepare_data(excel_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Determine the number of samples in the minority class
    class_counts = y_train.value_counts()
    minority_class_count = class_counts.min()

    # Adjust k_neighbors for SMOTE based on the number of samples in the minority class
    k_neighbors = min(minority_class_count - 1, 5)

    # Oversample the minority class using SMOTE
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()

    # Define the SVM model with RBF kernel and grid search for parameter tuning
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': [0.001, 0.01, 0.1, 1]
    }

    pipeline = make_pipeline(scaler, SVC(kernel='rbf'))
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    # Train the model using grid search
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Best parameters from grid search
    print("Best parameters:", grid_search.best_params_)

    # Predict and evaluate the model on the test set
    y_pred = grid_search.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Perform cross-validation on the balanced data
    cross_val_scores = cross_val_score(grid_search.best_estimator_, X_train_resampled, y_train_resampled, cv=5)
    print(f"Cross-validation accuracy scores: {cross_val_scores}")
    print(f"Mean cross-validation accuracy: {cross_val_scores.mean()}")

if __name__ == "__main__":
    # Path to your Excel file with ANOVA results
    excel_file_path = r'E:\ConnectivityMatrix\RM_ANOVA_results_with_interactions.xlsx'

    # Run the SVM analysis with class balancing and parameter tuning
    run_svm_analysis(excel_file_path)
