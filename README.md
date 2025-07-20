# Clinical Transcript Analyzer

Analyzes clinical transcripts to identify patients and extract medical entities using NER and embedding similarity.

## Design Approach

For this task, I chose HuggingFace's `d4data/biomedical-ner-all` NER model with ~30K downloads, specifically trained on medical data. The model extracts the key entities we need - medications, procedures, symptoms, and body parts. I leverage the model's contextual embeddings for both entity matching and patient name detection, creating a unified semantic space for medical terms.

The system performs both exact string matching and semantic similarity matching using cosine similarity on the embeddings. This captures semantic relationships like "knee pain" ↔ "osteoarthritis" that basic string matching would miss. Different entity types receive weighted importance scores (medications: 3.0x, symptoms: 2.0x, disease disorders: 2.5x, body parts: 1.5x, names: 4.0x) since exact medication matches and patient names are more reliable indicators than general symptoms.

The likelihood score represents matching confidence through a discriminative approach - positive evidence from entity matches combined with penalty scoring for contradictions (e.g., mentioning drugs a patient is allergic to). The system uses softmax normalization to convert raw scores into interpretable probabilities. Currently, we don't have ground truth accuracy metrics since this is unsupervised matching, but the likelihood scores serve as confidence estimates.

My main concern is false patient identification, which could be dangerous in clinical settings. The discriminative scoring helps, but for production I'd monitor precision scores and return multiple high-confidence matches for human review rather than forcing a single prediction. For storage, I'd migrate from CSV to PostgreSQL or DataBricks for scalability and reliability.

## Real-World Improvements

In a production system, I would implement several enhancements: 

**Advanced MLE approach:** Use maximum likelihood estimation based on patient demographics, medical history patterns, and symptom co-occurrence probabilities derived from large clinical datasets. This would provide more sophisticated priors than simple entity matching.

**Enhanced embedding strategies:** Beyond NER embeddings, I'd incorporate sentence-level and contextual embeddings to handle negation and context. For example, "you can't take ibuprofen" should match patients allergic to ibuprofen, but current entity-based matching might incorrectly favor patients who normally take ibuprofen.

## Running

**Start server:** `python api.py` → http://localhost:8000/

### Analyze Transcript
```bash
curl -X POST "http://0.0.0.0:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "transcript": "Nurse: Good morning, how are you feeling today? Patient: I am okay, but I had some trouble sleeping last night. Nurse: I am sorry to hear that. Was it the pain again? Patient: Yeah, my knee has been really stiff, especially when I try to get out of bed. Nurse: On a scale of 1 to 10, how bad is the pain this morning? Patient: Maybe a 6. It was worse yesterday. Nurse: Did the acetaminophen help at all? Patient: A little, but not much. I think it wore off pretty quickly. Nurse: Okay. I will let Dr. Levin know. We may try switching to ibuprofen if it is safe with your other meds. Any nausea or dizziness? Patient: No, just tired. And I have been feeling more short of breath when I walk to the bathroom. Nurse: Noted. We will check your oxygen levels as well."
     }'
```

### List Patients
```bash
curl -X GET "http://0.0.0.0:8000/patients"
```

### Add Patient
```bash
curl -X POST "http://0.0.0.0:8000/patients/add" \
     -H "Content-Type: application/json" \
     -d '{
       "patient": {
         "patient_id": "C",
         "name": "Robert Chen",
         "age": 72,
         "sex": "Male",
         "primary_diagnosis": "Osteoarthritis (Right Knee)",
         "secondary_conditions": "Mild Depression",
         "current_medications": "Acetaminophen (PRN), Sertraline daily",
         "history": "Right knee replacement surgery 4 months ago, Good recovery progress",
         "allergies": "Ibuprofen (severe allergic reaction), Aspirin (stomach bleeding)",
         "notes": "Robert was admitted for follow-up rehab on July 15th 2025 after his knee replacement. He has been making excellent progress with physical therapy and is eager to return to his daily walks. Patient educated on avoiding NSAIDs due to severe allergies."
       }
     }'
```

### Remove Patient
```bash
curl -X DELETE "http://0.0.0.0:8000/patients/C"
```
