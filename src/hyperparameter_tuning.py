import optuna
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV


def optimize_hyperparameters(model, param_grid, X_train, y_train, n_iter=20, cv=5, n_trials=50):

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_

    def objective_rf(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        predictions = rf_model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        return mse

    def objective_gb(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        gb_model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        gb_model.fit(X_train, y_train)
        predictions = gb_model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        return mse

    def objective_lr(trial):
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        predictions = lr_model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        return mse

    if isinstance(model, RandomForestRegressor):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_rf, n_trials=n_trials)
        best_params.update(study.best_params)
    elif isinstance(model, GradientBoostingRegressor):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_gb, n_trials=n_trials)
        best_params.update(study.best_params)
    elif isinstance(model, LinearRegression):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_lr, n_trials=n_trials)
        best_params.update(study.best_params)

    return best_params
