"""Example of CLA prediction POS.

"""

from collections import deque
import time

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.modelfactory import ModelFactory

import nltk
from nltk.corpus import brown

import model_params

def RunPOSPrediction():
  # Create the model for predicting POS.
  model = ModelFactory.create(model_params.MODEL_PARAMS)
  model.enableInference({'predictedField': 'pos'})
  # The shifter will align prediction and actual values.
  shifter = InferenceShifter()
  # Keep the last WINDOW predicted and actual values for plotting.
  #actHistory = deque([0.0] * WINDOW, maxlen=60)
  #predHistory = deque([0.0] * WINDOW, maxlen=60)

  # Initialize the plot lines that we will update with each new record.
  #actline, = plt.plot(range(WINDOW), actHistory)
  #predline, = plt.plot(range(WINDOW), predHistory)
  # Set the y-axis range.
  #actline.axes.set_ylim(0, 100)
  #predline.axes.set_ylim(0, 100)

  corpus = nltk.Text(word.lower() for word in brown.words())
  part_of_speech_list = nltk.pos_tag(corpus)

  for i, word_pos in enumerate(part_of_speech_list):
    s = time.time()

    # Get the CPU usage.
    #cpu = psutil.cpu_percent()
    word, pos = word_pos
    if i == len(part_of_speech_list) - 1:
      break

    next_word, next_pos = part_of_speech_list[i + 1]

    # Run the input through the model and shift the resulting prediction.
    modelInput = {'pos': pos}
    result = shifter.shift(model.run(modelInput))

    # Update the trailing predicted and actual value deques.
    inference = result.inferences['multiStepBestPredictions'][5]

    if inference is not None:
      print("Actual: %s/%s, Predicted: %s/%s" % (next_word, result.rawInput['pos'], "?", inference))
    #if inference is not None:
    #  actHistory.append(result.rawInput['cpu'])
    #  predHistory.append(inference)

    # Redraw the chart with the new data.
    #actline.set_ydata(actHistory)  # update the data
    #predline.set_ydata(predHistory)  # update the data
    #plt.draw()
    #plt.legend( ('actual','predicted') )    

    sleep(1)
    # Make sure we wait a total of 2 seconds per iteration.
    #try: 
    #  plt.pause(SECONDS_PER_STEP)
    #except:
    #  pass

if __name__ == "__main__":
  RunPOSPrediction()

