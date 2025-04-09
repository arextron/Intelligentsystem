import json
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple

class ChatbotEvaluator:
    def __init__(self, test_data_path: str, api_endpoint: str = "http://localhost:8000/chat"):
        self.api_endpoint = api_endpoint
        self.test_data = self._load_test_data(test_data_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.similarity_threshold = 0.6
    
    def _load_test_data(self, path: str) -> List[Dict]:
        with open(path, 'r') as f:
            data = json.load(f)
        return data['questions']
    
    def _get_chatbot_response(self, question: str) -> str:
        payload = {
            "user_input": question,
            "user_id": "evaluation_user"
        }
        try:
            print("requesting")
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            print("response received")
            return response.json()['response']
        except Exception as e:
            print(f"Error getting response for question '{question}': {e}")
            return ""
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.model.encode([text1, text2])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])  # Convert to native float
    
    def evaluate(self) -> Dict:
        results = {
            'total_questions': len(self.test_data),
            'correct_answers': 0,
            'category_accuracy': {},
            'similarity_scores': []
        }
        
        category_counts = {}
        
        for item in self.test_data:
            question = item['question']
            expected_answer = item['expected_answer']
            category = item['category']
            
            chatbot_response = self._get_chatbot_response(question)
            similarity = self._calculate_similarity(expected_answer, chatbot_response)
            results['similarity_scores'].append(float(similarity))  # Convert to native float
            
            is_correct = similarity >= self.similarity_threshold
            if is_correct:
                results['correct_answers'] += 1
            
            if category not in category_counts:
                category_counts[category] = {'correct': 0, 'total': 0}
            category_counts[category]['total'] += 1
            if is_correct:
                category_counts[category]['correct'] += 1
        
        results['overall_accuracy'] = float(results['correct_answers'] / results['total_questions'])
        results['average_similarity'] = float(np.mean(results['similarity_scores']))
        
        for category, counts in category_counts.items():
            results['category_accuracy'][category] = float(counts['correct'] / counts['total'])
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        report = [
            "Chatbot Performance Evaluation Report",
            "====================================",
            f"Total Questions Evaluated: {results['total_questions']}",
            f"Correct Answers: {results['correct_answers']}",
            f"Overall Accuracy: {results['overall_accuracy']:.2%}",
            f"Average Semantic Similarity: {results['average_similarity']:.2f}",
            "\nCategory-wise Performance:"
        ]
        
        for category, accuracy in results['category_accuracy'].items():
            report.append(f"- {category.capitalize()}: {accuracy:.2%}")
        
        report.extend([
            "\nEvaluation Criteria:",
            "- Answers considered correct if semantic similarity ≥ 0.7",
            "- Semantic similarity calculated using cosine similarity of sentence embeddings"
        ])
        
        return "\n".join(report)

if __name__ == "__main__":
    print("loading")
    evaluator = ChatbotEvaluator("benchmarkQs.json")
    results = evaluator.evaluate()
    report = evaluator.generate_report(results)
    print(report)
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)