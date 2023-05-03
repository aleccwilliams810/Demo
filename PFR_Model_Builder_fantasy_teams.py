import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

#create_model() creates a Keras model with specified hyperparameters
def create_model(input_shape, neurons=64, dropout_rate=0.1, optimizer='adam'):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neurons, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons // 2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def create_and_evaluate_model(pos, merged_data_scaled, param_grid, n_folds, n_features, dropout_rate=0.1):
    print(f'Creating model for position: {pos}')

    train_data = merged_data_scaled[(merged_data_scaled['Year'] < 2020) & (merged_data_scaled['FantPos'] == pos)]
    test_data = merged_data_scaled[(merged_data_scaled['Year'] < 2022) & (~merged_data_scaled.index.isin(train_data.index)) & (merged_data_scaled['FantPos'] == pos)]

    X_train = train_data.drop(['next_year_PPR', 'Year'], axis=1).select_dtypes(include=[np.number])
    y_train = train_data['next_year_PPR']
    X_test = test_data.drop(['next_year_PPR', 'Year'], axis=1).select_dtypes(include=[np.number])
    y_test = test_data['next_year_PPR']

    lin_reg = LinearRegression()
    rfe = RFE(lin_reg, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)

    feature_index = rfe.get_support(indices=True)
    print(f'Selected features for {pos}: {feature_index}')

    X_train = X_train.iloc[:, feature_index]
    X_test = X_test.iloc[:, feature_index]
    
    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)


    model = KerasRegressor(
        build_fn=create_model,
        input_shape=(n_features,),
        epochs=100,
        batch_size=10,
        verbose=0,
        dropout_rate = .1
    )
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_result = grid.fit(X_train, y_train)

    best_params = grid_result.best_params_
    best_score = -grid_result.best_score_

    print("Best Parameters:", best_params)
    print("Best MSE:", best_score)

    final_model = create_model(input_shape=(X_train.shape[1],), dropout_rate=best_params['dropout_rate'], optimizer=best_params['optimizer'])
    final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

    mse, mae = final_model.evaluate(X_test, y_test, verbose=0)
    print("Mean Squared Error for test data:", mse)
    print("Mean Absolute Error:", mae)

    return pos, final_model