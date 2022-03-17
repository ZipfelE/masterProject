###############################################################################################
# Die Datei "agentsScript.py" steuert den Lernprozess und verwaltet die DQNs der Agenten      #
# Nachdem alle benötigten Pakete importiert wurden, werden Objekte der Klasse "Agent" erzeugt #
# Die "main" Funktion ist eine Endlosschleife und wartet auf Daten der Simulationsagenten     #
# Sie wird wiederholt, bis der Abbruchsbefehl vom Simulationsmodell kommt                     #
# Die RL-Agenten haben verschiedene Aufgaben zu erfüllen, unterschieden durch If-Abfragen     #
###############################################################################################
# Pakete für Debugging
import logging
# Pakete für die TCP Verbindung zu Plant Simulation (Socket)
import socket
import struct
# Pakete für Berechnungen
import numpy as np 
import collections
import random
# Pakete für das DQN
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import TensorBoard
#import selbst programmierter Klassen
from dqn import *
from convdata import *
#######################################################################
# Alle Hyperparameter für das Training und für das neuronale Netz     #
# werden von Plant Simulation übergeben.                              #
# Funktionen des Agenten: "get_action", "train", "replay", "remember",#
#######################################################################
class Agent:
    def __init__(self, init_data):
        # Variablen der Umwelt
        tup = (int(init_data[0][0]),)
        self.observations = tup
        self.actions = int(init_data[1][0])

        # Variablen des Agenten
        self.replay_buffer_size = int(init_data[2][0])
        self.batch_size = int(init_data[3][0])
        self.train_start = int(init_data[4][0])
        self.epsilon = round(init_data[5][0], 4)
        self.epsilon_decay = round(init_data[6][0], 4)
        self.epsilon_min = round(init_data[7][0], 4)
        self.gamma = round(init_data[8][0], 4)
        self.memory = collections.deque(maxlen=self.replay_buffer_size)

        # DQN Variablen
        self.state_shape = self.observations
        self.learning_rate = init_data[9][0]
        #self.hid_layer = int(init_data[10][0])
        #was hid_layer now hid_layers as argument in DQN() --> change if necessary to go back
        self.hid_layers = init_data[10:][0]
        self.act_fkt = "relu"
        self.model = DQN(self.state_shape, self.actions, self.learning_rate, self.hid_layers, self.act_fkt) # erzeuge neuronales Netz (policy Netz)
        self.target_model = DQN(self.state_shape, self.actions, self.learning_rate, self.hid_layers, self.act_fkt) # 2. neuronales Netz für Experience Replay (Zielnetz)

        # Zusätzliche Variablen
        self.total_rewards = []
    
    # Funktion für die Auswahl einer Aktion Zufall/Vorhersage
    def get_action(self, state):
        # in epsilon Prozent der Fälle 
        if np.random.rand() <= self.epsilon:
            # gib eine Zufallszahl im Bereich aller Aktionen (0-3)
            return np.random.randint(self.actions), 0
        else:
            # hole eine vorhergesagte Aktion vom DQN-Model
            return np.argmax(self.model.predict(state)), 1
    
    # Funktion für die Trainingsschritte: 
    # speichere die gewählte Aktion "a" in Zustand "s" und die erhaltene Belohnung "r" endend in Zustand "S'" 
    def train(self, PSinput):
        sim_episode = 0.0
        done = bool(PSinput[2])
        # wandle erhaltene Daten um und speichere diese in Variablen
        state, action, reward, next_state, total_reward, sim_episode = convert.get_memory(PSinput)
        # formatiere Zustand für das Netzwerk um (8,1) -> (1,8)
        state = np.reshape(state, (1, state.shape[0]))
        next_state = np.reshape(next_state, (1, next_state.shape[0]))
        #sichere Variablen in den Erfahrungsspeicher (Replay Buffer)
        self.remember(state, action, reward, next_state, done)
        # rufe Funktion "replay" um von vorherigem Schritt zu lernen
        self.replay()

        if done:
            self.total_rewards.append(total_reward)
            mean_reward = np.mean(self.total_rewards[-10:])
            self.target_model.update_model(self.model)
            print(f"Agent: {agentIndex}, Episode: {sim_episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}, Ended: {done}, Datenmenge: {len(self.memory)}")
        

        #######################################################################

    def replay(self):
        # starte Training nur, wenn genügend Daten im Erfahrungsspeicher sind
        if len(self.memory) < self.train_start:
            return
        #ziehe einen Datenauszug aus dem Erfahrungsspeicher
        minibatch = random.sample(self.memory, self.batch_size)
        #speichere diesen in Listenvariablen
        states, actions, rewards, states_next, dones = zip(*minibatch)

        # [s1, s2, s3, s4, s5] ursprüngliches Listendesign
        # np.array([[s1], [s2], ...]) Design erwartet von Keras -> concatenate(numpy array)
        states = np.concatenate(states)
        states_next = np.concatenate(states_next)
        # Vorhersage aus dem live Netz für aktuelle Q-Werte
        # und Vorhersage aus dem Zielnetz für die Q-Werte der Folgezustände
        q_values = self.model.predict(states)
        q_values_next = self.target_model.predict(states_next) #q' und a' Werte
        #aktualisiere Q-Werte für die gerade ausgeführte Aktion des Auszugs
        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            # wenn Schritt zum Ende geführt hat, werden nur die unmittelbaren
            # Belohnungen berücksichtigt, ansonsten werden die Folgezustände
            # diskontiert und aufsummiert
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i], axis=0)

        # trainiere das Modell mit den Q-Werten und den Zuständen des Datenauszugs
        self.model.train(states, q_values)

    def remember(self, state, action, reward, next_state, done):
        # Füge aktuellen Zustand, Aktion, Belohnung, Folgezustand dem Überlaufspeicher hinzu
        self.memory.append((state, action, reward, next_state, done))
        # wende die epsilon greedy an: reduziere epsilon bei jedem Schritt 
        # bis epsilon_min erreicht ist
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
    
########################################################################### 
while True:
    try:
        if __name__ == "__main__":
            logger = logging.getLogger()
            # starte Verbindung mit Socket zum Datenempfang
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP
            server_addr = ("127.0.0.1", 30000)#hostname, port
            client_socket.connect(server_addr)
            init = True
            

            # Endlosschliefe zum Datenempfang, Datenverarbeitung und erledigen der Aufgaben
            while True:
                # sende eine Nachricht an PlantSim bei der Initialisierung
                if init == True:
                    init = False
                    # erzeuge ein dictionary "agents"
                    agents = {}
                    #kommunikation zu plantsimulation
                    client_socket.send(bytes("0/"+"t/"+"connection enabled/", "utf8"))
                    print("connection enabled")

                # Socket auf Empfang, wartend auf Dateneingänge
                PSinput = client_socket.recv(1024)
                
                # Teste die Verbindung bevor Daten gesendet/empfangen werden
                if len(PSinput) <= 0:
                    break
                
                # erzeuge ein Objekt "convert" von der Klasse ConvData
                convert = ConvData(PSinput)
                # extrahiere den Index des aufrufenden Agenten
                agentIndex = convert.get_agentIndex(PSinput)
                # wähle zugehörigen agent aus dem dictionary
                agent = agents.get(agentIndex)
                # extrahiere die Aufgabe aus dem PlantSim Input

                task = convert.get_task(PSinput)

                ####################### initialisiere den Agenten #############     
                if task == 1:
                    # extrahiere nur die Initialisierungsdaten von PSinput
                    data = convert.get_init_data(PSinput)
                    #erzeuge ein Objekt der Klasse "Agent" mit den übergebenen Daten
                    agent = Agent(data)
                    # erzeuge einen Eintrag in agents mit dem key agentIndex und als Wert agent
                    agents[agentIndex] = agent

                    # sende Nachricht an PlantSim (zur Fortführung in PlantSim)
                    client_socket.send(bytes(str(agentIndex)+"/"+"t/"+"Agent initialized/", "utf8"))
                    print(f"Agent {agentIndex} initialized") # Drucke in der Konsole die Nachricht
                ###############################################################
                
                ### speichere Daten in Erfahrungsspeicher und trainiere Netz ##
                elif task == 2:
                    agent.train(PSinput)
                ###############################################################

                ########### hole Aktion (Zufall/Vorhersage) von Agent #########
                elif task == 3:
                    # extrahiere Zustand aus Daten
                    state = convert.get_data(PSinput)
                    # umformen des Tupel von (8,1) nach (1,8)
                    state = np.reshape(state, (1, state.shape[0]))
                    # hole Aktion von Agent
                    action, rp = agent.get_action(state)
                    # sende Aktion an PlantSim ("a" für Aufgabe in PlantSim, "/" als Endmarker für Nachricht)
                    client_socket.send(bytes(str(agentIndex)+"/"+"a/"+str(action)+"/"+str(rp)+"/", "utf8"))
                ################################################################

                ########### hole Aktion (Vorhersage) vom gleadenen Netz ########
                elif task == 4:
                    # extrahiere Zustand aus Daten
                    state = convert.get_data(PSinput)
                    # umformen des Tupel von (8,1) nach (1,8)
                    state = np.reshape(state, (1, state.shape[0]))
                    # hole Aktion von Agent
                    action = np.argmax(agent.model.predict(state))
                    rp = 1
                    # sende Aktion an PlantSim ("a" als Aufgabe in PlantSim, "/" als Endmarker für Nachricht
                    client_socket.send(bytes(str(agentIndex)+"/"+"a/"+str(action)+"/"+str(rp)+"/", "utf8"))
                ################################################################

                ########### sichere Agentennetz (model) ########################
                elif task == 5:
                    path = PSinput
                    agent.model.save_model("C:/Users/elias/Documents/Master-SYI/Masterprojekt/training-arena/ReinforcedLearningEnvironment/savedDQNs/dqn_PlantSimAgent"+str(agentIndex)+".h5")
                    print(f"Model of agent {agentIndex} saved")
                    # sende Nachricht an PlantSim
                    client_socket.send(bytes(str(agentIndex) + "/" + "t/" + "DQN Model saved/", "utf8"))
                ################################################################

                ########### lade Agentennetz (model) ###########################
                elif task == 6:
                    path = PSinput
                    agent.model.load_model("C:/Users/elias/Documents/Master-SYI/Masterprojekt/training-arena/ReinforcedLearningEnvironment/savedDQNs/dqn_PlantSimAgent"+str(agentIndex)+".h5")
                    print(f"Model for agent {agentIndex} loaded")
                    # sende Nachricht an PlantSim
                    client_socket.send(bytes(str(agentIndex) + "/" + "t/" + "DQN Model loaded/", "utf8"))
                ################################################################
                
                #### beende Socket-Client Verbindung zum Simulationsagenten ####
                elif task == 8:
                    break
                ################################################################

            ################### schließe Client-Socket #########################
            if task == 8:
                # sende Nachricht an PlantSim zum beenden der Verbindung
                client_socket.send(bytes(str(agentIndex)+"/"+"t/"+"connection to socket successfully closed!/", "utf8"))
                # schließe das Socket
                client_socket.close()
                break





            ####################################################################
    except Exception:
        logger.error("Fatal error in main loop", exc_info=True)

