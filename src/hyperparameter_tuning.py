from sklearn.model_selection import RandomizedSearchCV

def optimize_hyperparameters(model, param_grid, X, y, n_iter=20, cv=5):
    random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=cv, n_jobs=-1, random_state=42)
    random_search.fit(X, y)
    return random_search.best_params_
