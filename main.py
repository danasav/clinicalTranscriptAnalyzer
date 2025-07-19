# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# Load model directly
from medical_ner_processor import MedicalNERProcessor


def main():
    model_name = "d4data/biomedical-ner-all" # 30K

    # Your examples
    text1 = """Transcript (Nurse-Patient Conversation)
                Nurse: Good morning, how are you feeling today?
                Patient: I'm okay, but I had some trouble sleeping last night.
                Nurse: I'm sorry to hear that. Was it the pain again?
                Patient: Yeah, my knee has been really stiff, especially when I try to get out of bed.
                Nurse: On a scale of 1 to 10, how bad is the pain this morning?
                Patient: Maybe a 6. It was worse yesterday.
                Nurse: Did the acetaminophen help at all?
                Patient: A little, but not much. I think it wore off pretty quickly.
                Nurse: Okay. I’ll let Dr. Levin know. We may try switching to ibuprofen if it’s safe with your other meds. Any nausea or dizziness?
                Patient: No, just tired. And I’ve been feeling more short of breath when I walk to the bathroom.
                Nurse: Noted. We’ll check your oxygen levels as well."""

    cls = MedicalNERProcessor()
    print(cls.get_entities(text1))
    print(cls.get_embeddings(text1))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
