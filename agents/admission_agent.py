class AdmissionAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.admission_data = {
            "deadlines": "2025-01-15",
            "requirements": "GPA 3.0+, CS prerequisites",
            "program_info": "4-year Computer Science program..."
        }
        
    def handle_query(self, query, context):
        # Search admission data first
        for key in self.admission_data:
            if key in query.lower():
                return self.admission_data[key]
        # Fallback to vector store context
        return f"Admission Info: {self.admission_data.get('general', 'Contact admissions@concordia.ca')}"