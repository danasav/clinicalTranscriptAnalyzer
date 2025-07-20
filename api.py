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
    uvicorn.run(app, host="0.0.0.0", port=8000)