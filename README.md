# masterProject

*Production control using self-learning multi-agent-systems*

*Implementation using a virtual simulation model*

*year: 2021 - 2022*

## Verwendete Software:

- Plant Simulation 16.0.3 inkl. Interface Package (für Socket)
- Anaconda 3.8 -> virtuelle Environment "AnacondaENV.yml" laden 
  (dort sind alle benötigten Pakete installiert)


## notwendige Pfadanpassungen:

- im PlantSimulation-Modell im Objekt "OpenPython" den Pfad zur Datei "openAgent.cmd" eintragen
- in "openAgent.cmd" den Pfad für den Ordner "LernenderAgent" korrigieren
- in "agent.py" den Pfad unter task = 6 zum Laden der vortrainierten KNN korrigieren


## Schritte zur Simulation mit der Logik DeepRL - Trainingsmodus:

- "start socket network"
- öffnen des Dialogs der DeepRL-Agenten (innerhalb der Agenten-Netzwerke)
- "create RL-Agent (DQN)"
- kein Haken bei "Prediction without Training"
- Sowohl die Konsole in PlantSimulation als auch die Shell sollten jetzt die Initialisierung des jeweiligen Agenten ausgegeben
- sind alle Agenten initialisiert kann die Simulation gestartet werden


## Schritte zur Simulation mit der Logik DeepRL - vortrainierte KNN:

- "start socket network"
- öffnen des Dialogs der DeepRL-Agenten (innerhalb der Agenten-Netzwerke)
- "create RL-Agent (DQN)"
- "load existing DQN"
- Haken bei "Prediction without Training"
- Sowohl die Konsole in PlantSimulation als auch die Shell sollten jetzt die Initialisierung und das Laden des jeweiligen Agenten ausgegeben
- sind alle KNN geladen kann die Simulation gestartet werden