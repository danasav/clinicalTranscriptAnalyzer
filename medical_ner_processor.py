import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class MedicalNERProcessor:
    def __init__(self):
        """
        Initialize the NER processor with the biomedical model
        """
        model_name = "d4data/biomedical-ner-all"  # 30K downloaded
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model = self.model.cpu()  # I run it locally without a GPU :(

        self.ner_pipeline = pipeline(task="ner",
                                     model=self.model,
                                     tokenizer=self.tokenizer,
                                     aggregation_strategy="simple",
                                     device=-1)

        # Storage for processed entities
        self.entity_store = pd.DataFrame(columns=[
            'text', 'entity_text', 'entity_type', 'start_pos', 'end_pos',
            'context', 'confidence', 'embedding'
        ])

    def get_entities(self, text: str) -> pd.DataFrame:
        """
        Extract entities using the NER pipeline and convert to DataFrame
        """
        entities_df = pd.DataFrame.from_records(self.ner_pipeline(text))
        self.entity_store = pd.concat([self.entity_store, entities_df],
                                      ignore_index=True)
        return entities_df

    def get_embeddings(self, text: str) -> (np.ndarray, np.ndarray):
        """
        Extract the latent space's embeddings from the text
        """
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping")

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]  # Last layer embeddings
        return hidden_states[0], offset_mapping[0]

    def extract_entity_embeddings(self, text: str) -> pd.DataFrame:
        """
        Extract entities and their embeddings in the NER latent space from the text
        """
        entities_df = self.get_entities(text)
        if entities_df.empty:
            # incase there aren't any entities, there is nothing to extract
            return pd.DataFrame()

        hidden_states, offset_mapping = self.get_embeddings(text)

        def get_entity_embedding(row: pd.Series) -> np.ndarray:
            """
            Get the embedding for a single entity
            """
            entity_start, entity_end = row['start'], row['end']

            token_starts = offset_mapping[:, 0]
            token_ends = offset_mapping[:, 1]

            # Find overlapping tokens for the entity
            overlap_mask = (token_starts < entity_end) & (token_ends > entity_start)
            overlapping_indices = torch.where(overlap_mask)[0]

            if len(overlapping_indices) > 0:
                return hidden_states[overlapping_indices].mean(dim=0).numpy()
            else:
                return np.zeros(hidden_states.shape[-1])

        entities_df['embedding'] = entities_df.apply(get_entity_embedding, axis=1)
        return entities_df
