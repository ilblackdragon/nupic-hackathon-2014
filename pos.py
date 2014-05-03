
import nltk


def PartOfSpeechTagging(sentence):
  """
  >>> PartOfSpeechTagging("And now for something completely different.")
  [('And', 'CC'), ('now', 'RB'), ('for', 'IN'), ('something', 'NN'), ('completely', 'RB'), ('different', 'JJ')]
  """
  text = nltk.word_tokenize("And now for something completely different")
  pos_tags = nltk.pos_tag(text)
  return pos_tags 


if __name__ == "__main__":
  pos_tags = PartOfSpeechTagging("And now for something completely different.")
  print(pos_tags)
