import json

translate_dict = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "dog": "cane", "cavallo": "horse", "elephant" : "elefante", "butterfly": "farfalla", "chicken": "gallina", "cat": "gatto", "cow": "mucca","ragno":"spider", "spider": "ragno", "squirrel": "scoiattolo"}

if __name__ == "__main__":
    json.dump(translate_dict, open("translate.json","w"))