TEST_IMAGES_DIR = '/home/rumaxx/road-signs-project/ml/data'
TEST_CSV_DIR = '/home/rumaxx/road-signs-project/ml/data/Test.csv'
MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/custom_cnn_150_model.h5'
HISTORY_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/custom_cnn_150_history.json'

NUM_CATEGORIES = 43

CLASSES = {
    0: 'Ograniczenie prędkości (20km/h)', 1: 'Ograniczenie prędkości (30km/h)', 
    2: 'Ograniczenie prędkości (50km/h)', 3: 'Ograniczenie prędkości (60km/h)', 
    4: "Ograniczenie prędkości (70km/h)", 5: "Ograniczenie prędkości (80km/h)", 
    6: "Koniec ograniczenia prędkości (80km/h)", 7: "Ograniczenie prędkości (100km/h)", 
    8: "Ograniczenie prędkości (120km/h)", 9: "Zakaz wyprzedzania", 
    10: "Zakaz wyprzedzania przez pojazdy ciężarowe", 11: "Skrzyżowanie z drogą podporządkowaną", 
    12: "Droga z pierwszeństwem", 13: "Ustąp pierwszeństwa", 14: "Stop", 
    15: "Zakaz ruchu", 16: "Zakaz wjazdu pojazdów ciężarowych", 17: "Zakaz wjazdu", 
    18: "Inne niebezpieczeństwo", 19: "Niebezpieczny zakręt w lewo", 
    20: "Niebezpieczny zakręt w prawo", 21: "Podwójny zakręt, pierwszy w lewo", 
    22: "Nierówna droga", 23: "Śliska jezdnia", 24: "Zagrożenie zwężeniem jezdni - prawostronne", 
    25: "Roboty drogowe", 26: "Sygnalizacja świetlna", 27: "Przejście dla pieszych", 
    28: "Dzieci", 29: "Rowerzyści", 30: "Oszronienie jezdni", 
    31: "Dzikie zwierzęta", 32: "Koniec zakazów", 33: "Nakaz jazdy w prawo", 
    34: "Nakaz jazdy w lewo", 35: "Nakaz jazdy prosto", 36: "Nakaz jazdy prosto lub w prawo", 
    37: "Nakaz jazdy prosto lub w lewo", 38: "Nakaz jazdy z prawej strony znaku", 
    39: "Nakaz jazdy z lewej strony znaku", 40: "Rondo", 
    41: "Koniec zakazu wyprzedzania", 42: "Koniec zakazu wyprzedzania przez pojazdy ciężarowe",
}
