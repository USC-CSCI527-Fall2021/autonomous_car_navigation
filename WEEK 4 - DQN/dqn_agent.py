from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random


MIN_REPLAY_MEMORY_SIZE = 500
MINIBATCH_SIZE = 128
PREDICTION_BATCH_SIZE = MINIBATCH_SIZE
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4

class DQN_AGENT():
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights( self.model.get_weights() )
        self.target_update_counter = 0
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
    
    #change to dueling architecture
    def create_model(self):
        base_model = VGG16(weights=None, include_top = False, input_shape=(84, 84, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs= base_model.input, outputs=predictions)
        model.compile( loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy", ] )
        return model
    
    def train(self, replay_memory):
        if len(replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        #PRIORTIZED Experience replay to be implemented here
        minibatch = random.sample(replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])        
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
        
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)
        
        X, y = [], []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)


        self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False)
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]