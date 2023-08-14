import numpy as np
import samediff
from scipy.spatial.distance import pdist
# import data_processing as dp
import glob
# import random
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

def eval(X, matches):
    print("Entered eval: ")
    distances = pdist(X, 'cosine')

    # ap = average_precision_score(np.asarray(f1_corres_matches).astype(int), distances)
    # print("average_precision_score: ", ap)
    
    ap, prb = samediff.average_precision(distances[matches == True], distances[matches == False], show_plot=True)

    print("Average precision:", ap, flush=True)
    print("Precision-recall breakeven:", prb, flush=True)


def GetValidData(data_path):
    train_sets = glob.glob(data_path + "test_words.t*")
    contents = []
    for item in train_sets:
        contents += open(item, 'r').read().splitlines()
    print("lenght of contents: ", len(contents))
    # contents = random.sample(contents, 500)
    iv = ['carpet', 'maybe', 'comes', 'always', 'beyond', 'played', 'though', 'people', 'anything', 'enemy', 'first', 'beautiful', 'later', 'without', 'nature', 'fruit', 'whole', 'common', 'marriage', 'useful', 'existed', 'children', 'brown', 'seemed', 'economic', 'family', 'possible', 'surface', 'halloween', 'important', 'longer', 'garbage', 'small', 'front', 'steve', 'based', 'could', 'meeting', 'novel', 'completely', 'things', 'position', 'every', 'others', 'provides', 'class', 'began', 'study', 'contains', 'would', 'water', 'organizations', 'single', 'right', 'added', 'morning', 'greasy', 'popularity', 'intelligence', 'particularly', 'general', 'toward', 'occasionally', 'allow', 'question', 'available', 'increases', 'often', 'large', 'shellfish', 'artists', 'certain', 'still', 'teeth', 'fresh', 'problem', 'years', 'trish', 'shelter', 'quick', 'little', 'ambulance', 'thing', 'requires', 'never', 'street', 'indeed', 'slipped', 'pressure', 'great', 'treatment', 'slowly', 'formula', 'steep', 'death', 'showed', 'related', 'range', 'enough', 'house', 'attention', 'higher', 'changes', 'black', 'getting', 'cloth', 'forces', 'money', 'woman', 'paper', 'white', 'withdraw', 'annoying', 'sound', 'eight', 'subway', 'enjoy', 'birth', 'lines', 'clearly', 'child', 'activity', 'development', 'already', 'effects', 'followed', 'table', 'brief', 'ralph', 'voice', 'number', 'results', 'either', 'answer', 'musicians', 'close', 'answered', 'serve', 'government', 'skirt', 'bright', 'relief', 'experience', 'several', 'charge', 'heavy', 'project', 'degree', 'broken', 'autumn', 'program', 'needed', 'leaves', 'young', 'ideas', 'change', 'necessary', 'hours', 'however', 'destroy', 'coincided', 'system', 'increased', 'youngsters', 'ocean', 'problems', 'order', 'tomorrow', 'bring', 'nothing', 'thick', 'perhaps', 'within', 'sleeping', 'style', 'theme', 'military', 'yellow', 'garage', 'coming', 'world', 'another', 'cases', 'mother', 'honey', 'gives', 'analysis', 'ingredients', 'history', 'various', 'words', 'informative', 'reflect', 'simple', 'times', 'learn', 'makes', 'shoes', 'around', 'something', 'become', 'means', 'nobody', 'story', 'better', 'sense', 'across', 'strength', 'lower', 'cream', 'place', 'eating', 'strong', 'looked', 'ability', 'working', 'thought', 'quite', 'carry', 'events', 'extra', 'control', 'please', 'along', 'three', 'clumsy', 'shown']
    oov = ['twilight', 'weatherproof', 'untimely', 'january', 'papered', 'calcium', 'superb', 'somewhat', 'robin', 'fairy', 'synagogue', 'accusations', 'approval', 'parenthood', 'pretty', 'arrange', 'tropical', 'materials', 'technology', 'shoulder', 'oriental', 'matched', 'perfume', 'autistic', 'diseases', 'overlooked', 'barometric', 'walking', 'believe', 'blouses', 'different', 'cornered', 'beach', 'swung', 'december', 'abdomen', 'misquote', 'bleachers', 'industry', 'aluminum', 'substances', 'kindergarten', 'unusual', 'flood', 'elderly', 'school', 'luxurious', 'audition', 'expected', 'livestock', 'behavior', 'pewter', 'improving', 'geese', 'cutbacks', 'scarf', 'living', 'thursdays', 'sounded', 'argued', 'slope', 'employment', 'dislikes', 'bones', 'ended', 'especially', 'soothed', 'fortune', 'chosen', 'scared', 'income', 'wheel', 'stray', 'swedish', 'prime', 'happening', 'journal', 'iguanas', 'marvelously', 'centrifuge', 'cleans', 'convenient', 'emblem', 'medical', 'kidnappers', 'phony', 'months', 'victim', 'special', 'welfare', 'generous', 'artificial', 'society', 'fallout', 'detailed', 'countryside', 'cleaned', 'shore', 'oasis', 'customer', 'chablis', 'seattle', 'rather', 'antelope', 'aches', 'lawyers', 'accomplished', 'highway', 'spend', 'noise', 'laugh', 'gallon', 'combine', 'unlimited', 'consists', 'thinner', 'retracted', 'mergers', 'overcharged', 'remote', 'forms', 'assistance', 'sweater', 'cheap', 'burned', 'diagram', 'errors', 'overalls', 'michael', 'medieval', 'grows', 'tooth', 'grades', 'objects', 'frost', 'squeaked', 'stockings', 'stake', 'entertaining', 'planned', 'purpose', 'cashmere', 'neglect', 'spinach', 'acropolis', 'cranberry', 'production', 'aglow', 'electron', 'catastrophic', 'account', 'exposure', 'reads', 'dirty', 'cleaners', 'millionaires', 'supervision', 'shirt', 'chest', 'solve', 'framework', 'weekday', 'excess', 'promote', 'balls', 'unevenly', 'excluded', 'nightly', 'ankle', 'crayons', 'depicts', 'basketball', 'authorized', 'grandmother', 'business', 'heating', 'bedroom', 'rescue', 'galoshes', 'freeway', 'pleasantly', 'products', 'shampooed', 'prevented', 'magnetic', 'target', 'fashion', 'massage', 'purchase', 'swing', 'refused', 'farmyard', 'written', 'stole', 'prospective', 'sugar', 'brochure', 'reasons', 'smiles', 'rationalize', 'appetizers', 'boring', 'items', 'dishes', 'coleslaw', 'parental', 'avoid', 'sampling', 'beggar', 'endurance', 'proceeding', 'lessons', 'contagious', 'hostages', 'lighted', 'preparation', 'woolen', 'tragic', 'hierarchies', 'leadership', 'shaving', 'controlled', 'personnel', 'magnetism', 'lived', 'abruptly', 'wondered', 'plenty', 'aside', 'tradition', 'miraculously', 'songs', 'obvious', 'guard', 'literature', 'elegant', 'chief', 'gentleman', 'sketched', 'sprained', 'bugle', 'hurts', 'agricultural', 'situated', 'meats', 'noteworthy', 'gunpoint', 'distance', 'twelfth', 'decorate', 'popular', 'symbols', 'cliff', 'silly', 'looking', 'breakdown', 'holidays', 'goals', 'gunman', 'silverware', 'human', 'expense', 'nearest', 'carol', 'leeway', 'stopwatch', 'driving', 'potatoes', 'expensive', 'horizon', 'block', 'reptiles', 'frantically', 'treats', 'favor', 'worry', 'costume', 'guess', 'stranded', 'viewpoint', 'straw', 'bungalow', 'bonfire', 'electrical', 'outgrew', 'fleecy', 'healthier', 'spilled', 'boots', 'stupid', 'lunch', 'arriving', 'shame', 'skirts', 'muscular', 'distributed', 'machine', 'vocabulary', 'strongly', 'short', 'contributory', 'alligators', 'prestige', 'petticoats', 'upbringing', 'salesmanship', 'review', 'colored', 'murals', 'exercise', 'sport', 'ambled', 'candy', 'wardrobe', 'pizzerias', 'corduroy', 'spider', 'forgot', 'diversity', 'generals', 'causeway', 'gloves', 'miami', 'butcher', 'flimsy', 'harmonize', 'dance', 'cutting', 'contain', 'heels', 'solid', 'thighs', 'obtain', 'whenever', 'mirage', 'glistened', 'buyer', 'audits', 'classrooms', 'frame', 'needlepoint', 'orders', 'poultry', 'apology']

    iv_positions = []
    oov_positions = []

    x2 = []
    c2 = []
    for item in contents:
        npy_path = "_".join(item.split("_")[1:]).replace(".wav", ".npy")
        word = item.split("_")[0]
        x2.append(npy_path)
        c2.append(word)
    matches2 = samediff.generate_matches_array(c2)

    x3 = []
    c3 = []
    pos3 = []
    for pos, item in enumerate(contents):
        npy_path = "_".join(item.split("_")[1:]).replace(".wav", ".npy")
        word = item.split("_")[0]
        if word in iv:
            x3.append(npy_path)
            c3.append(word)
            pos3.append(pos)
    matches3 = samediff.generate_matches_array(c3)

    x4 = []
    c4 = []
    pos4 = []
    for pos, item in enumerate(contents):
        npy_path = "_".join(item.split("_")[1:]).replace(".wav", ".npy")
        word = item.split("_")[0]
        if word in oov:
            x4.append(npy_path)
            c4.append(word)
            pos4.append(pos)
    matches4 = samediff.generate_matches_array(c4)

    print(len(pos3), len(pos4))
    return matches2, matches3, matches4, pos3, pos4


def main():
    count = 3988 
    # count = 50
    embeddings = []
    for i in range(count):
        embeddings.append(np.mean(np.load("/GeneralizedKWS/testing_output/embeddings/embeddings1/emb1_" + str(i) + ".npy"), axis=1).squeeze())
    print("Shape of the embeddings1: ", np.asarray(embeddings).shape)

    matches2, matches3, matches4, pos3, pos4 = GetValidData('/GeneralizedKWS/evaluation_codes/multiview/Data/timit/')

    print("ALL EMBEDDINGS: ")
    eval(embeddings, matches2)

    print("IV EMBEDDINGS: ")
    eval(np.asarray(embeddings)[pos3], matches3)

    print("OOV EMBEDDINGS: ")
    eval(np.asarray(embeddings)[pos4], matches4)


if __name__ == "__main__":
    main()
    
