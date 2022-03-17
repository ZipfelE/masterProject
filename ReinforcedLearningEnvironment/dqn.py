
from keras.models import *
from keras.layers import *
from keras.optimizers import *



class DQN(Model): #in Klammern übergeben an Instanzen von Klasse
    def __init__(self, state_shape, num_actions, lr, hid_layers, act_fkt):
        #aufruf des Constructors der Oberklasse
        super(DQN, self).__init__() #für die Vererbung notwendig (target Netzwerk) 
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr
        self.hid_layers = hid_layers
        self.act_fkt = act_fkt #relu
        num_hid_layers = len(self.hid_layers)

        # Festlegen der Inputschicht
        state = Input(shape=self.state_shape)
        x = state
        #Festlegen der versteckten Schichten
        for i in range(num_hid_layers):
            num_neurons = int(self.hid_layers[i])
            x = Dense(num_neurons)(x)
            x = Activation(act_fkt)(x)
        #Festlegen der Ausgangsschicht
        out = Dense(self.num_actions)(x)
        self.model = Model(inputs=state, outputs=out)
        self.model.compile(loss="mse", optimizer=Adam(lr=self.lr))


    def train(self, states, q_values):#, tensorboard):
        self.model.fit(states, q_values, verbose=0)#, callbacks=[tensorboard])

    def predict(self, states):
        return self.model.predict(states)

    def update_model(self, other_model):
        self.model.set_weights(other_model.get_weights())

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path)