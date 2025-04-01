# utils/external_api.py
import wikipedia

class WikipediaAPI:
    def __init__(self):
        wikipedia.set_lang("en")
        
    def search(self, query):
        try:
            return wikipedia.summary(query, sentences=2, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=2)
        except wikipedia.exceptions.PageError:
            return "No relevant Wikipedia information found."
        except Exception as e:
            return f"Error fetching Wikipedia data: {str(e)}"