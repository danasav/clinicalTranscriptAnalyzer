
from medical_ner_processor import MedicalNERProcessor
from patient_matcher import PatientMatcher
import pandas as pd


class ClinicalTranscriptAnalyzer:
    """
    Encapsulates all the clinical analysis functionalities:
        - Data management
        - Medical entity extraction
        - Patient matching to a given clinical transcript
    """
    def __init__(self):
        """
        Initialize ClinicalTranscriptAnalyzer
        """
        self.patients_csv_path = 'Data/patients_data.csv'
        self.ner_processor = MedicalNERProcessor()
        self.patient_matcher = PatientMatcher()
        self.patients_df = self.load_patients_data()
        self.patient_entity_table = self.create_patient_entity_table()

    def load_patients_data(self) -> pd.DataFrame:
        """
        Load patients data
        """
        return pd.read_csv(self.patients_csv_path)

    def save_patients_data(self) -> None:
        """
        Save current patients data
        """
        self.patients_df.to_csv(self.patients_csv_path, index=False)

    def create_patient_entity_table(self) -> pd.DataFrame:
        """
        Process patient DataFrame and extract all entities into tabular format
        Returns DataFrame with one row per entity
        """
        patients_data = self.patients_df.set_index('patient_id')
        text_columns = ['primary_diagnosis', 'secondary_conditions',
                        'current_medications', 'history', 'allergies', 'notes']
        all_patient_entities = []

        for patient_id, patient_row in patients_data.iterrows():
            for col in text_columns:
                entities_df = self.ner_processor.extract_entity_embeddings(patient_row[col])
                if not entities_df.empty:
                    entities_df['patient_id'] = patient_id
                    entities_df['source_field'] = col
                    all_patient_entities.append(entities_df)
        return pd.concat(all_patient_entities, ignore_index=True)

    def refresh_patient_entity_table(self):
        """
        Refresh the patient entity table after data changes
        """
        self.patient_entity_table = self.create_patient_entity_table()

    def extract_medical_entities(self, transcript_text: str) -> pd.DataFrame:
        """
        Extract diagnoses, medications, and treatments/procedures from transcript and output as JSON
        """
        entities_df = self.ner_processor.extract_entity_embeddings(transcript_text).drop(columns=['embedding'])

        entity_mapping = {
            'Disease_disorder': 'diagnosis',
            'Sign_symptom': 'diagnosis',
            'Medication': 'medication',
            'Diagnostic_procedure': 'procedure',
            'Therapeutic_procedure': 'procedure'
        }

        entities_df['category'] = entities_df['entity_group'].replace(entity_mapping)
        res_df = entities_df[entities_df['category'].isin(['diagnosis', 'medication', 'procedure'])].drop(columns=['entity_group'])
        return res_df

    def analyze_transcript(self, transcript: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Main analysis method - given a transcript, returns patient match and medical entities
        """
        transcript_entity_table = self.ner_processor.extract_entity_embeddings(transcript)
        results_df, matches_df, penalties_df = self.patient_matcher.compute_discriminative_likelihood(
            transcript_entity_table, self.patient_entity_table, self.patients_df, transcript
        )

        best_match = results_df.loc[0, ['patient_id', 'likelihood_score', 'num_matches', 'penalty_score']]
        best_match['explanation'] = f"Matched based on {int(best_match['num_matches'])} entity matches with {float(best_match['penalty_score'])} penalty points"

        medical_entities_df = self.extract_medical_entities(transcript)

        return (best_match[['patient_id', 'likelihood_score', 'explanation']].to_frame().T,
                medical_entities_df)

    def add_patient(self, patient_data: dict) -> pd.DataFrame:
        """
        Add a single patient.
        Patient is given as a dict with patient parameters
        """
        if patient_data['patient_id'] in self.patients_df['patient_id'].values:
            raise ValueError(f"Patient ID {patient_data['patient_id']} already exists")

        new_patient_df = pd.DataFrame([patient_data])
        self.patients_df = pd.concat([self.patients_df, new_patient_df], ignore_index=True)
        self.save_patients_data()
        self.refresh_patient_entity_table()

        return len(self.patients_df)

    def remove_patient(self, patient_id: str) -> int:
        """
        Remove a patient
        """
        if patient_id not in self.patients_df['patient_id'].values:
            raise ValueError(f"Patient {patient_id} not found")

        self.patients_df = self.patients_df[self.patients_df['patient_id'] != patient_id]
        self.save_patients_data()
        self.refresh_patient_entity_table()

        return len(self.patients_df)

    def get_patients_summary(self) -> pd.DataFrame:
        """
        Get summary of all patients
        """
        return self.patients_df[['patient_id', 'name', 'age', 'primary_diagnosis']]
