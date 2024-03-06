**Dual Attention based Bi-directional LSTM using Bayesian Optimization**

\# Set your features and target variable

features = \[\'Open\', \'High\', \'Low\', \'Close\'\] \# Add other
relevant features

target = \'Close\'

\# Split the data into train, validation, and test sets

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3,
random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
test_size=0.5, random_state=42)

\# Define the model architecture

def create_model(params):

model = Sequential()

\# Encoder

model.add(Bidirectional(LSTM(params\[\'lstm_units\'\],
return_sequences=True), input_shape=(params\[\'sequence_length\'\],
len(features))))

model.add(Attention(use_scale=True))

\# Decoder

model.add(LSTM(params\[\'lstm_units\'\], return_sequences=True))

model.add(Attention(use_scale=True))

model.add(Dense(1, activation=\'linear\'))

optimizer = Adam(learning_rate=params\[\'learning_rate\'\])

model.compile(optimizer=optimizer, loss=\'mean_squared_error\')

return model

\# Define the search space for Bayesian optimization

param_space = {

\'lstm_units\': (10, 100),

\'sequence_length\': (10, 30),

\'learning_rate\': (1e-4, 1e-2)

}

\# Define the model and search for the best hyperparameters

model = create_model

search = BayesSearchCV(model, param_space, n_iter=50, cv=3, n_jobs=-1)

search.fit(X_train, y_train)

\# Get the best hyperparameters

best_params = search.best_params\_

\# Train the model with the best hyperparameters

final_model = create_model(best_params)

final_model.fit(X_train, y_train, epochs=100, batch_size=32,
validation_data=(X_val, y_val),
callbacks=\[EarlyStopping(patience=10)\])

\# Evaluate the model on the test set

y_pred = final_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f\'Mean Absolute Error on Test Set: {mae}\')
