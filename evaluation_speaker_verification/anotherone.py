import nemo.collections.asr as nemo_asr
from scipy.spatial.distance import cosine

def calculate_cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)  # Returns similarity in range [0, 1]

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_small')
vector1 = speaker_model.get_embedding(r"C:\Users\hp\Downloads\final_try_3.wav")
vector2 = speaker_model.get_embedding(r"C:\Users\hp\Downloads\final_try_7.wav")

print(calculate_cosine_similarity(vector1[0], vector2[0]))

