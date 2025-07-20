from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

from clinical_transcript_analyzer import ClinicalTranscriptAnalyzer


# Request/Response models
class AnalyzeRequest(BaseModel):
    transcript: str


class PatientRecord(BaseModel):
    patient_id: str
    name: str
    age: int
    sex: str
    primary_diagnosis: str
    secondary_conditions: str
    current_medications: str
    history: str
    allergies: str
    notes: str


class AddPatientRequest(BaseModel):
    patient: PatientRecord


class AnalyzeResponse(BaseModel):
    patient_match: Dict[str, Any]
    medical_entities: Dict[str, Any]


# Initialize FastAPI app
app = FastAPI(title="Clinical Transcript Analyzer", version="1.0.0")

# Single instance of our analyzer
analyzer = ClinicalTranscriptAnalyzer()


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_transcript(request: AnalyzeRequest):
    """
    Analyze clinical transcript
    """
    try:
        patient_match, medical_entities_df = analyzer.analyze_transcript(request.transcript)
        medical_entities = {
            "diagnosis": medical_entities_df[medical_entities_df['category'] == 'diagnosis'].to_dict('records'),
            "medication": medical_entities_df[medical_entities_df['category'] == 'medication'].to_dict('records'),
            "procedure": medical_entities_df[medical_entities_df['category'] == 'procedure'].to_dict('records')
        }
        return AnalyzeResponse(
            patient_match=patient_match.to_dict('records')[0],
            medical_entities=medical_entities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/patients/add")
async def add_patient(request: AddPatientRequest):
    """
    Add a new patient
    """
    try:
        total_patients = analyzer.add_patient(request.patient.dict())
        return {"message": f"Patient {request.patient.patient_id} added successfully",
                "total_patients": total_patients}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add patient: {str(e)}")


@app.delete("/patients/{patient_id}")
async def remove_patient(patient_id: str) -> Dict[str, Any]:
    """
    Remove a patient
    """
    try:
        total_patients = analyzer.remove_patient(patient_id)
        return {"message": f"Patient {patient_id} removed successfully",
                "total_patients": total_patients}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove patient: {str(e)}")


@app.get("/patients")
async def list_patients() -> Dict[str, Any]:
    """
    Get list of all patients
    """
    patients_summary = analyzer.get_patients_summary().to_dict('records')
    return {"patients": patients_summary, "total_count": len(patients_summary)}


@app.get("/")
async def root():
    return {"message": "Clinical Transcript Analyzer API", "analyze": "/analyze"}


if __name__ == "__main__":
    # print(analyzer.analyze_transcript("""Nurse: Good morning, how are you feeling today?
    # Patient: I'm okay, but I had some trouble sleeping last night.
    # Nurse: I'm sorry to hear that. Was it the pain again?
    # Patient: Yeah, my knee has been really stiff, especially when I try to get out of bed.
    # Nurse: On a scale of 1 to 10, how bad is the pain this morning?
    # Patient: Maybe a 6. It was worse yesterday.
    # Nurse: Did the acetaminophen help at all?
    # Patient: A little, but not much. I think it wore off pretty quickly.
    # Nurse: Okay. I’ll let Dr. Levin know. We may try switching to ibuprofen if it’s safe with your other meds. Any nausea or dizziness?
    # Patient: No, just tired. And I’ve been feeling more short of breath when I walk to the bathroom.
    # Nurse: Noted. We’ll check your oxygen levels as well.""")[0].to_dict('records'))
    # # print(analyzer.create_patient_entity_table().drop(columns=['embedding']).to_dict('records'))
    #
    # print(analyzer.analyze_transcript("My knee has been stiff. Did the acetaminophen help?")[0].to_dict('records'))
    # print(analyzer.analyze_transcript(""""Nurse: Good morning Robert, how are you feeling today?
    # Patient: Much better actually! My knee replacement recovery has been going really well.
    #  Nurse: That is wonderful to hear. Any pain in the knee today?
    #  Patient: Just a little stiffness when I first get up, but the acetaminophen helps.
    #  Nurse: Great. Have you been doing your physical therapy exercises?
    #  Patient: Yes, every day. I am actually looking forward to getting back to my daily walks soon.
    #  Nurse: Excellent progress. Any issues with your mood lately?
    #  Patient: I have been feeling much more positive since starting the sertraline.
    #  My depression seems well controlled. Nurse: That is good news.
    #  Any side effects from the antidepressant? Patient: None that I have noticed.
    #  I am just glad to be feeling like myself again.""")[0].to_dict('records'))

    uvicorn.run(app, host="0.0.0.0", port=8000)