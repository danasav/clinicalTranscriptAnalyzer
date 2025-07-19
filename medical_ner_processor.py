import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class MedicalNERProcessor:
    def __init__(self):
        """Initialize the NER processor with the biomedical model"""
        model_name = "d4data/biomedical-ner-all"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

        # Storage for processed entities
        self.entity_store = pd.DataFrame(columns=[
            'text', 'entity_text', 'entity_type', 'start_pos', 'end_pos',
            'context', 'confidence', 'embedding'
        ])

    def get_entities(self, text, context_window=20):
        """Extract entities using the NER pipeline and convert to DataFrame"""
        # Get entities from pipeline
        entities_df = pd.DataFrame.from_records(self.ner_pipeline(text))

        self.entity_store = pd.concat([self.entity_store, entities_df], ignore_index=True)
        return entities_df

    def get_embeddings(self, text):
        """Extract the latent space's embeddings from the text"""
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]  # Last layer embeddings
        return hidden_states, inputs['offset_mapping']