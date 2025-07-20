from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re


class PatientMatcher:
    """
    Finding matches between transcript and patient entities
    """
    def __init__(self):
        # Entity type importance weights
        self.entity_weights = {
            'Medication': 4.0,
            'Sign_symptom': 2.0,
            'Biological_structure': 1.5,
            'Diagnostic_procedure': 1.0,
            'Disease_disorder': 2.5,
            'PERSON_NAME': 4.0
        }

    def find_name_matches(self, transcript_text: str, patient_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find patient name matches in transcript using regex
        Returns DataFrame with name matches
        """
        name_matches = []

        for _, patient in patient_df.iterrows():
            patient_id = patient['patient_id']
            patient_name = patient['name']

            name_parts = patient_name.split()

            for name_part in name_parts:
                pattern = r'\b' + re.escape(name_part) + r'\b'
                matches_found = re.finditer(pattern, transcript_text, re.IGNORECASE)

                for match in matches_found:
                    name_matches.append({
                        'transcript_entity': match.group(),
                        'patient_entity': name_part,
                        'entity_type': 'PERSON_NAME',
                        'weight': 4.0,
                        'score': 1.0,
                        'patient_id': patient_id,
                        'source_field': 'name',
                        'start': match.start(),
                        'end': match.end()
                    })

        return pd.DataFrame(name_matches)

    def compute_exact_matches(self, transcript_df: pd.DataFrame, patient_df: pd.DataFrame) -> pd.DataFrame:
        """
        Find exact matches between transcript NER entities and patient NER entities
        """
        transcript_df['word'] = transcript_df['word'].str.lower().str.strip()
        patient_df['word'] = patient_df['word'].str.lower().str.strip()

        matches = transcript_df.merge(
            patient_df,
            on=['word', 'entity_group'],
            suffixes=('_transcript', '_patient')
        )

        if matches.empty:
            return pd.DataFrame(columns=['transcript_entity', 'patient_entity', 'entity_type',
                                         'weight', 'score', 'patient_id', 'source_field'])

        # Add weights and scores
        matches['weight'] = matches['entity_group'].map(self.entity_weights).fillna(1.0)
        matches['score'] = 1.0

        return matches

    def compute_embedding_similarities(self, transcript_entities: pd.DataFrame, patient_entities: pd.DataFrame, threshold: float=0.7) -> pd.DataFrame:
        """
        Find high similarity matches between transcript and patient entities, by the embedded vectors
        """

        all_similarities = []
        for entity_type in transcript_entities['entity_group'].unique():
            t_entities = transcript_entities[transcript_entities['entity_group'] == entity_type].reset_index(drop=True)
            p_entities = patient_entities[patient_entities['entity_group'] == entity_type].reset_index(drop=True)

            if len(t_entities) == 0 or len(p_entities) == 0:
                continue

            similarity_matrix = cosine_similarity(np.vstack(t_entities['embedding'].values),
                                                  np.vstack(p_entities['embedding'].values))
            high_sim_indices = np.where(similarity_matrix >= threshold)

            if len(high_sim_indices[0]) > 0:
                t_indices = high_sim_indices[0]
                p_indices = high_sim_indices[1]

                matches_df = pd.DataFrame({
                    'transcript_entity': t_entities.iloc[t_indices]['word'].values,
                    'patient_entity': p_entities.iloc[p_indices]['word'].values,
                    'entity_type': entity_type,
                    'weight': self.entity_weights.get(entity_type, 1.0),
                    'score': similarity_matrix[t_indices, p_indices],
                    'patient_id': p_entities.iloc[p_indices]['patient_id'].values,
                    'source_field': p_entities.iloc[p_indices]['source_field'].values
                })
                all_similarities.append(matches_df)

        if all_similarities:
            return pd.concat(all_similarities, ignore_index=True)
        else:
            return pd.DataFrame(columns=['transcript_entity', 'patient_entity', 'entity_type',
                                         'weight', 'score', 'patient_id', 'source_field'])

    def compute_patient_likelihood(self,
                                   transcript_entities: pd.DataFrame,
                                   patient_entity_table: pd.DataFrame,
                                   patient_df: pd.DataFrame,
                                   transcript_text: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Compute likelihood scores for each patient including name matching
        """
        exact_matches = self.compute_exact_matches(transcript_entities, patient_entity_table)
        semantic_matches = self.compute_embedding_similarities(transcript_entities, patient_entity_table)
        name_matches = self.find_name_matches(transcript_text, patient_df)

        all_matches_list = [exact_matches, semantic_matches]
        if not name_matches.empty:
            all_matches_list.append(name_matches)

        all_matches = pd.concat(all_matches_list, ignore_index=True) if all_matches_list else pd.DataFrame()
        if all_matches.empty:
            return (all_matches.assign(likelihood_score=0, raw_score=0, num_matches=0),
                    all_matches)

        patient_scores = all_matches.groupby('patient_id').apply(
            lambda group: (group['weight'] * group['score']).sum() / len(transcript_entities)
        ).to_dict()
        # Convert to probabilities using softmax
        scores_array = np.array(list(patient_scores.values()))
        if scores_array.sum() > 0:
            probabilities = np.exp(scores_array) / np.sum(np.exp(scores_array))
        else:
            probabilities = np.ones(len(scores_array)) / len(scores_array)

        results_df = pd.DataFrame({
            'patient_id': list(patient_scores.keys()),
            'likelihood_score': probabilities,
            'raw_score': list(patient_scores.values())
        })

        match_counts = all_matches.groupby('patient_id').size().to_dict()
        results_df['num_matches'] = results_df['patient_id'].map(match_counts).fillna(0)
        # Sort by likelihood
        results_df = results_df.sort_values('likelihood_score', ascending=False).reset_index(drop=True)

        return results_df, all_matches

    def compute_discriminative_likelihood(self,
                                          transcript_entities: pd.DataFrame,
                                          patient_entity_table: pd.DataFrame,
                                          patient_df: pd.DataFrame,
                                          transcript_text: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Enhanced likelihood computation with negative evidence penalties.
        Currently only implemented for medications and allergies, but can be extended to other entity types
        """
        results_df, all_matches = self.compute_patient_likelihood(
            transcript_entities, patient_entity_table, patient_df, transcript_text
        )

        all_penalties = []
        for patient_id in patient_entity_table['patient_id'].unique():
            patient_entities = patient_entity_table[
                patient_entity_table['patient_id'] == patient_id
                ].copy()

            allergy_meds = patient_entities[
                (patient_entities['source_field'] == 'allergies') &
                (patient_entities['entity_group'] == 'Medication')
                ]['word'].str.lower().str.strip()

            transcript_meds = transcript_entities[
                transcript_entities['entity_group'] == 'Medication'
                ]['word'].str.lower().str.strip()

            for med in transcript_meds:
                for allergy in allergy_meds:
                    if (pd.notna(allergy) and
                            allergy != 'none reported' and
                            (med in allergy or allergy in med)):
                        all_penalties.append({
                            'patient_id': patient_id,
                            'type': 'allergy_contradiction',
                            'transcript_entity': med,
                            'patient_entity': allergy,
                            'penalty': 3.0
                        })

        if all_penalties:
            penalties_df = pd.DataFrame(all_penalties)
            patient_penalties = penalties_df.groupby('patient_id')['penalty'].sum().to_dict()
        else:
            penalties_df = pd.DataFrame()
            patient_penalties = {}

        results_df['penalty_score'] = results_df['patient_id'].map(patient_penalties).fillna(0)
        results_df['discriminative_raw_score'] = results_df['raw_score'] - results_df['penalty_score']

        disc_scores = np.maximum(results_df['discriminative_raw_score'].values, 0.001)

        if disc_scores.sum() > 0:
            discriminative_probs = np.exp(disc_scores) / np.sum(np.exp(disc_scores))
        else:
            discriminative_probs = np.ones(len(disc_scores)) / len(disc_scores)

        results_df['discriminative_likelihood'] = discriminative_probs
        results_df = results_df.sort_values('discriminative_likelihood', ascending=False).reset_index(drop=True)

        return results_df, all_matches, penalties_df