from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import logging
import time

class Modelling:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def tune(self, model, param_grid, cv=3, scoring=None):
        # Perform grid search for hyperparameter tuning
        tuned_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
        # Fit the model on training set
        start = time.time()
        tuned_model.fit(self.X_train, self.y_train)
        end = time.time()
        logging.info(f"Training took {end - start:.2f} seconds")
        
        # Print the best parameters and the best score
        logging.info("Fitted model with hyperparameter tuning")
        logging.info(f"Best Parameters: {tuned_model.best_params_}")
        logging.info(f"Best Cross-Validation Score: {tuned_model.best_score_}")
        return tuned_model.best_estimator_
    
    def train(self, model, param_tune=False, param_grid=None, cv=3, scoring=None):
        # Fit & train the model
        model = model
        if param_tune==False:
            start = time.time()
            trained_model = model.fit(self.X_train, self.y_train)
            end = time.time()
            logging.info(f"Training took {end - start:.2f} seconds")
            logging.info("Fitted model without hyperparameter tuning")
            logging.info(trained_model)
        elif param_tune==True:
            trained_model = self.tune(model, param_grid, cv, scoring)
        else:
            raise ValueError("param_tune must be either True or False.")
        return trained_model

    def predict(self, model, pred_data=None, true_label=None):
        # Predict using the orginal train and test sets
        if pred_data is None:
            logging.info("Predicting using the original train and test sets...")
            y_pred = model.predict(self.X_train)
            true_label = self.y_train
    
            logging.info(f"R² Score on train set: {r2_score(true_label, y_pred):.4f}")
            logging.info(f"MAE: {mean_absolute_error(true_label, y_pred):.4f}")
            logging.info(f"RMSE: {mean_squared_error(true_label, y_pred):.4f}")
            y_pred = model.predict(self.X_test)
            true_label = self.y_test
            logging.info(f"R² Score on test set: {r2_score(true_label, y_pred):.4f}")
            logging.info(f"MAE: {mean_absolute_error(true_label, y_pred):.4f}")
            logging.info(f"RMSE: {mean_squared_error(true_label, y_pred):.4f}")
        
        # Predict using the provided pred_data and actual_data
        elif pred_data is not None and true_label is not None:
            logging.info("Predicting using provided pred_data and actual_data...")
            y_pred = model.predict(pred_data)
            logging.info("R² Score on provided label set:", r2_score(true_label, y_pred))
        
        # Raise error
        else:
            raise ValueError("Either pred_data and actual_data must be provided or both must be None.")

    def save_model(self, model, path):
        import joblib
        joblib.dump(model, path) # Save the trained model to the specified path
        logging.info(f"Model saved to {path}")
    
    def load_model(self, path):
        import joblib
        return joblib.load(path)